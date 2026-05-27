import math
import huggingface_hub as hub
from tqdm import tqdm
from peft import PeftConfig, load_peft_weights, AutoPeftModel
from collections import defaultdict, OrderedDict
import torch
import os
from transformers import AutoConfig
from utils.misc_utils import check_nan_in_pytree
import json
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed


def check_shape_compatibility(
    state_dict: dict[str, torch.Tensor],
    target_module_info: OrderedDict[str, tuple[int, int]],
) -> bool:
    for key, shape in target_module_info.items():
        lora_A_key = f"{key}.lora_A.weight"
        lora_B_key = f"{key}.lora_B.weight"
        key_xor = int(lora_A_key in state_dict) + int(lora_B_key in state_dict)
        if key_xor == 0:
            continue
        elif key_xor == 1:
            return False
        elif key_xor == 2:
            if state_dict[lora_B_key].shape[0] != shape[0]:
                return False
            if state_dict[lora_A_key].shape[1] != shape[1]:
                return False
    return True


def _scan_single_model(
    model_id: str, target_base_model_name: str
) -> tuple[str, str | None]:
    """Scan a single model and return (model_id, result_category).

    result_category is one of: "config success", "not safetensors", "not causal lm lora",
    "not supported features", or None (for non-matching base model or exceptions).
    """
    try:
        peft_config = get_peft_config(model_id)
        if peft_config is None:
            return model_id, None
        base_model = peft_config.base_model_name_or_path

        if (
            base_model is None
            or target_base_model_name.split("/")[-1].lower() not in base_model.lower()
        ):
            return model_id, None

        # we only use .safetensors models
        if "adapter_model.safetensors" not in hub.list_repo_files(model_id):
            return model_id, "not safetensors"

        if (
            peft_config.peft_type != "LORA"
            or peft_config.modules_to_save is not None
            or peft_config.task_type != "CAUSAL_LM"
        ):
            return model_id, "not causal lm lora"

        # check lora config
        if (
            peft_config.bias != "none"
            or peft_config.use_dora
            or peft_config.layer_replication
            or peft_config.lora_bias
            or peft_config.corda_config
            or peft_config.eva_config
            or peft_config.loftq_config
            or peft_config.trainable_token_indices
            or peft_config.rank_pattern
            or peft_config.alpha_pattern
        ):
            return model_id, "not supported features"

        return model_id, "config success"

    except Exception as e:
        return model_id, f"exception {e.__class__.__name__}"


def collect_lora_model_ids(
    target_base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    output_path: str = "outputs/model_ids.txt",
    existing_model_id_file: str = "",
    overwrite: bool = False,
    raise_error: bool = False,
    num_workers: int = 16,
):
    if os.path.exists(output_path) and not overwrite:
        print(f"Model IDs file {output_path} already exists. Skipping collection.")
        with open(output_path, "r") as f:
            model_ids = [line.strip() for line in f.readlines()]
        return model_ids
    elif os.path.exists(output_path) and overwrite:
        print(
            f"Model IDs file {output_path} already exists, but overwrite is True. Rerunning collection process."
        )

    reason_counts = defaultdict(list)
    model_norms = {}
    checkpoint_path = output_path.replace(".txt", "_checkpoint.json")

    # Check for existing checkpoint (resume from scan)
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}, resuming from scan results...")
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)
        reason_counts = defaultdict(list, checkpoint_data.get("reason_counts", {}))
        model_norms = checkpoint_data.get("model_norms", {})
        processed_models = set(checkpoint_data.get("processed_models", []))
        print(
            f"Loaded {len(reason_counts['config success'])} models to download, {len(processed_models)} already processed"
        )
    else:
        processed_models = set()

        # collect model configs
        if existing_model_id_file:
            with open(existing_model_id_file, "r") as f:
                model_ids = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(model_ids)} model IDs from {existing_model_id_file}")
        else:
            api = hub.HfApi()
            # Filter by base_model:adapter tag to reduce API calls
            base_model_filter = f"base_model:adapter:{target_base_model_name}"
            print(f"Querying HuggingFace Hub with filter: {base_model_filter}")
            model_ids = (
                model.id
                for model in api.list_models(
                    fetch_config=True,
                    library="peft",
                    filter=base_model_filter,
                )
            )
            print(f"Loading model IDs from Hugging Face Hub (filtered by base_model)")

        # Convert generator to list for parallel processing
        if not isinstance(model_ids, list):
            print("Converting model IDs iterator to list...")
            model_ids = list(model_ids)
            print(f"Found {len(model_ids)} total PEFT models to scan")

        # Parallel config scanning
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _scan_single_model, model_id, target_base_model_name
                ): model_id
                for model_id in model_ids
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Scan model ids"
            ):
                model_id, result = future.result()
                if result is not None:
                    reason_counts[result].append(model_id)
                    if result == "config success":
                        reason_counts["base model matches"].append(model_id)

        # Save checkpoint after scan
        print(f"Saving scan checkpoint to {checkpoint_path}...")
        with open(checkpoint_path, "w") as f:
            json.dump(
                {
                    "reason_counts": dict(reason_counts),
                    "model_norms": model_norms,
                    "processed_models": list(processed_models),
                },
                f,
            )
        print(
            f"Checkpoint saved. {len(reason_counts['config success'])} models to download."
        )

    # collect model weights
    to_download = [
        m for m in reason_counts["config success"] if m not in processed_models
    ][::-1]
    print(f"Starting weight download for {len(to_download)} models...")
    download_failed_count = defaultdict(int)
    checkpoint_interval = 10  # Save checkpoint every N models
    models_since_checkpoint = 0

    while len(to_download) > 0:
        model_id = to_download.pop()
        state_dict = get_normalized_weights(model_id)
        peft_config = get_peft_config(model_id)
        if len(state_dict) == 0 or peft_config is None:
            download_failed_count[model_id] += 1
            if download_failed_count[model_id] > 3:
                print(f"Skipping {model_id} because it failed to download 3 times")
                reason_counts["download failed"].insert(0, model_id)
                processed_models.add(model_id)
                continue
            else:
                to_download.insert(0, model_id)
            continue

        # nan check
        has_nan = check_nan_in_pytree(state_dict, f"state_dict from {model_id}")
        if has_nan:
            print(f"Skipping {model_id} because it contains NaN")
            reason_counts["nan weights"].append(model_id)
            processed_models.add(model_id)
            continue

        # shape compatibility check
        target_module_shapes = get_target_module_info(target_base_model_name)
        shape_conflict = not check_shape_compatibility(state_dict, target_module_shapes)

        shape_conflict = False
        for key, shape in target_module_shapes.items():
            lora_A_key = f"{key}.lora_A.weight"
            lora_B_key = f"{key}.lora_B.weight"
            key_xor = int(lora_A_key in state_dict) + int(lora_B_key in state_dict)
            if key_xor == 0:
                # target module not selected in the model
                continue
            elif key_xor == 1:
                # target module selected in the model, but only one of the two lora matrices is present
                shape_conflict = True
                break
            elif key_xor == 2:
                # target module selected in the model, and both of the two lora matrices are present
                if state_dict[lora_B_key].shape[0] != shape[0]:
                    shape_conflict = True
                    break
                if state_dict[lora_A_key].shape[1] != shape[1]:
                    shape_conflict = True
                    break
                if state_dict[lora_A_key].shape[0] != state_dict[lora_B_key].shape[1]:
                    shape_conflict = True
                    break

        if shape_conflict:
            print(
                f"Skipping {model_id} because it has a shape conflict with the base model"
            )
            reason_counts["shape conflict"].append(model_id)
            processed_models.add(model_id)
            continue

        # lora norm check
        lora_norm_squared = 0
        for key, value in state_dict.items():
            if "lora_A" in key:
                lora_A = value
                lora_B = state_dict[key.replace("lora_A", "lora_B")]
                lora_norm_squared += (lora_B @ lora_A).norm() ** 2

        alpha = (
            peft_config.lora_alpha
            if isinstance(peft_config.lora_alpha, (float, int))
            else 8.0
        )
        r = (
            peft_config.r
            if isinstance(peft_config.r, int) and peft_config.r > 0
            else lora_A.shape[0]
        )
        if r <= 0:
            print(f"Skipping {model_id} because it has invalid rank {r}")
            reason_counts["invalid rank"].append(model_id)
            processed_models.add(model_id)
            continue

        rank = r**0.5 if peft_config.use_rslora else r
        lora_scale_factor = alpha / rank
        lora_norm = (lora_norm_squared**0.5) * lora_scale_factor

        if lora_norm <= 1.0 or lora_norm > 1000.0 or torch.isnan(lora_norm).any():
            print(f"Skipping {model_id} because it has an extreme lora norm")
            reason_counts["lora norm"].append(model_id)
            processed_models.add(model_id)
            continue

        print(f"Adding {model_id} with lora norm {lora_norm} to final list")
        reason_counts["success"].append(model_id)
        model_norms[model_id] = lora_norm.item()

        # Track processed and save checkpoint periodically
        processed_models.add(model_id)
        models_since_checkpoint += 1
        if models_since_checkpoint >= checkpoint_interval:
            with open(checkpoint_path, "w") as f:
                json.dump(
                    {
                        "reason_counts": dict(reason_counts),
                        "model_norms": model_norms,
                        "processed_models": list(processed_models),
                    },
                    f,
                )
            models_since_checkpoint = 0

    # Clean up checkpoint after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    model_ids = reason_counts["success"]
    with open(output_path, "w") as f:
        for model_id in model_ids:
            f.write(model_id + "\n")
    with open(output_path.replace(".txt", "_details.json"), "w") as f:
        json.dump(reason_counts, f, indent=4)
    with open(output_path.replace(".txt", "_norms.json"), "w") as f:
        json.dump(model_norms, f, indent=4)
    print(f"Dumped model IDs to {output_path}")

    return model_ids


def get_peft_model(model_id: str) -> AutoPeftModel | None:
    try:
        return AutoPeftModel.from_pretrained(model_id, trust_remote_code=False)
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return None


def get_normalized_weights(
    model_id_or_path: str,
    device: str | None = None,
    apply_alpha: bool = False,
    raise_error=False,
    subset_layers: List[int] = [],
) -> dict[str, torch.Tensor]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if apply_alpha:
        peft_config = get_peft_config(model_id_or_path)
        if peft_config is None:
            print(f"Error loading PEFT config for {model_id_or_path}")
            if raise_error:
                raise ValueError(f"Could not load PEFT config for {model_id_or_path}")
            return {}

    try:
        if model_id_or_path.endswith(".pt") and os.path.exists(model_id_or_path):
            state_dict = torch.load(model_id_or_path, map_location=device)
        else:
            state_dict = load_peft_weights(model_id_or_path, device=device)
    except Exception as e:
        print(f"Error loading model {model_id_or_path}: {e}")
        if raise_error:
            raise e
        else:
            return {}

    output_state_dict = {}
    for key, value in state_dict.items():
        if ".layers." in key:
            if (len(subset_layers) == 0) or (
                int(key.split(".layers.")[-1].split(".")[0]) in subset_layers
            ):
                output_state_dict[".layers." + key.split(".layers.", 1)[1]] = (
                    value.bfloat16()
                )
    del state_dict

    if apply_alpha:
        scaling_factor = (
            peft_config.lora_alpha / math.sqrt(peft_config.r)
            if peft_config.use_rslora
            else peft_config.lora_alpha / peft_config.r
        )
        assert scaling_factor > 0, f"Scaling factor is {scaling_factor} for {model}"
        output_state_dict = {
            key: (value * (scaling_factor**0.5))
            for key, value in output_state_dict.items()
        }

    return output_state_dict


def get_peft_config(model_id: str) -> PeftConfig | None:
    try:
        return PeftConfig.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return None


def get_target_module_info(
    target_base_model_name: str,
) -> OrderedDict[str, tuple[int, int]]:
    config = AutoConfig.from_pretrained(target_base_model_name)

    # Check for supported model architectures
    is_llama = "llama" in target_base_model_name.lower()
    is_qwen = "qwen" in target_base_model_name.lower()

    if is_llama or is_qwen:
        # Both Llama and Qwen use the same module structure
        module_dict = OrderedDict()

        # Get head_dim (may be explicit or computed)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        for layer_idx in range(config.num_hidden_layers):
            module_dict[f".layers.{layer_idx}.mlp.up_proj"] = (
                config.intermediate_size,
                config.hidden_size,
            )
            module_dict[f".layers.{layer_idx}.mlp.down_proj"] = (
                config.hidden_size,
                config.intermediate_size,
            )
            module_dict[f".layers.{layer_idx}.mlp.gate_proj"] = (
                config.intermediate_size,
                config.hidden_size,
            )
            module_dict[f".layers.{layer_idx}.self_attn.q_proj"] = (
                config.num_attention_heads * head_dim,
                config.hidden_size,
            )
            module_dict[f".layers.{layer_idx}.self_attn.k_proj"] = (
                config.num_key_value_heads * head_dim,
                config.hidden_size,
            )
            module_dict[f".layers.{layer_idx}.self_attn.v_proj"] = (
                config.num_key_value_heads * head_dim,
                config.hidden_size,
            )
            module_dict[f".layers.{layer_idx}.self_attn.o_proj"] = (
                config.hidden_size,
                config.num_attention_heads * head_dim,
            )

        return module_dict
    else:
        raise ValueError(f"Target base model {target_base_model_name} not supported")


if __name__ == "__main__":
    collect_lora_model_ids(
        # target_base_model_name="meta-llama/Llama-3.1-8B-Instruct",
        target_base_model_name="Qwen/Qwen3-4B-Instruct-2507",
        # output_path="results/model_lists/refiltered_model_ids.txt",
        output_path="results/model_lists/qwen4b_model_ids.txt",
        overwrite=True,
        raise_error=False,
    )

# Run
"""
python weight_processing/collect_lora_experts.py
"""
# Collect results at outputs/model_ids_new.txt, outputs/model_ids_new_details.json, outputs/model_ids_new_norms.json
