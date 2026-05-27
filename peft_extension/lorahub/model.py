from glob import glob
import torch
import os
import torch.nn as nn
from peft import PolyModel
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from .config import LorahubConfig
from .layers import LorahubLinear, LorahubModule
from tqdm import tqdm
from weight_processing.collect_lora_experts import (
    get_normalized_weights,
    get_peft_config,
)


def load_expert_weights(lora_module_list):
    """Load experts from huggingface and save the state_dict and peft_config"""

    cache = {}
    for peft_model_id in tqdm(lora_module_list):
        cache[peft_model_id] = {}
        print("> Loading {} ...".format(peft_model_id))

        # Use get_normalized_weights instead of loading full base model
        state_dict = get_normalized_weights(peft_model_id, device="cpu")
        peft_config = get_peft_config(peft_model_id)

        if len(state_dict) == 0 or peft_config is None:
            print(f"Failed to load {peft_model_id}, skipping...")
            del cache[peft_model_id]
            continue

        cache[peft_model_id]["state_dict"] = state_dict
        cache[peft_model_id]["peft_config"] = peft_config

    return cache


def concat_experts(cache):
    """
    1. concat the experts and 2. delimit where each expert starts and ends.
    """

    expert_list = list(cache.keys())
    concat_expert = {}

    all_targets = []
    for exp in cache:
        exp_layers = list(cache[exp]["state_dict"].keys())
        all_targets += exp_layers
    all_targets = set(all_targets)

    for layer in all_targets:
        start_idx = 0
        end_idx = None
        concat_expert[layer] = {"state_dict": None, "index": []}
        for exp in expert_list:
            # if this expert doesn't have this layer
            if layer not in cache[exp]["state_dict"]:
                end_idx = start_idx
            # if this expert indeed has this layer
            else:
                end_idx = start_idx + cache[exp]["peft_config"].r
                if concat_expert[layer]["state_dict"] == None:
                    concat_expert[layer]["state_dict"] = cache[exp]["state_dict"][layer]
                else:
                    if "lora_A" in layer:
                        concat_expert[layer]["state_dict"] = torch.concat(
                            (
                                concat_expert[layer]["state_dict"],
                                cache[exp]["state_dict"][layer],
                            ),
                            axis=0,
                        )
                    elif "lora_B" in layer:
                        concat_expert[layer]["state_dict"] = torch.concat(
                            (
                                concat_expert[layer]["state_dict"],
                                cache[exp]["state_dict"][layer],
                            ),
                            axis=1,
                        )

            concat_expert[layer]["index"].append((start_idx, end_idx))
            start_idx = end_idx

    return concat_expert, expert_list


class LorahubModel(BaseTuner):
    prefix: str = "poly_"
    # Boilerplate methods
    # It's hard to understand why PEFT didn't implement these methods in the parent class
    get_peft_config_as_dict = PolyModel.get_peft_config_as_dict
    enable_adapter_layers = PolyModel.enable_adapter_layers
    disable_adapter_layers = PolyModel.disable_adapter_layers

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the base model."""
        if not hasattr(self.model, "gradient_checkpointing_enable"):
            raise ValueError(
                f"The model {self.model.__class__.__name__} does not support gradient checkpointing"
            )
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on the base model."""
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        """Get input embeddings from the base model."""
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        """Get output embeddings from the base model."""
        return self.model.get_output_embeddings()

    # Model specific methods
    def __init__(self, model, peft_config, adapter_name: str = "default"):

        self.all_expert_info = self._load_expert_weight_and_metadata(
            peft_config.expert_info_dir, peft_config.expert_list_path
        )
        peft_config.rank = self.all_expert_info["rank"]
        peft_config.scaling = self.all_expert_info["scaling"]
        peft_config.num_experts = len(self.all_expert_info["rank"])
        peft_config.target_modules = self.all_expert_info["target_modules"]
        peft_config.expert_list = self.all_expert_info["expert_list"]
        self.train_lora_weights = peft_config.pi_tuning

        super().__init__(model, peft_config, adapter_name)

        if peft_config.weight_granularity == "per_expert":
            mapping_fn = lambda x: 0
        elif peft_config.weight_granularity == "per_layer":
            mapping_fn = lambda x: (x.split(".layers.")[-1]).split(".")[0]
        elif peft_config.weight_granularity == "per_sublayer":
            mapping_fn = lambda x: tuple((x.split(".layers.")[-1]).split(".")[:2])
        elif (
            peft_config.weight_granularity == "per_module"
            or peft_config.weight_granularity == "per_dimension"
        ):
            mapping_fn = None
        else:
            raise ValueError(
                f"Unsupported weight granularity: {peft_config.weight_granularity}"
            )

        if mapping_fn is not None:
            representative_modules = {}
            count = {}
            for module_name, module in self.named_modules():
                if isinstance(module, LorahubModule):
                    print(f"Processing module: {module_name}")
                    sharing_key = mapping_fn(module_name)
                    if sharing_key not in representative_modules:
                        representative_modules[sharing_key] = module
                        count[sharing_key] = 1
                    else:
                        module.poly_shared_expert_weights = representative_modules[
                            sharing_key
                        ].poly_shared_expert_weights
                        count[sharing_key] += 1
            print(
                f"Tied expert weights among {len(representative_modules)} groups based on {peft_config.weight_granularity}"
            )
            print(f"Tied key and counts: {list(count.items())}")
        else:
            print(f"No tied parameters for expert weights.")

        # required by the parent classes
        base_config = model.config
        self.config = base_config
        self.generation_config = model.generation_config

    def _load_expert_weight_and_metadata(
        self, expert_info_dir: str, expert_list_path: str, verbose: bool = True
    ):
        """Loads all expert state_dict concatenated at the layer-level and the corresponding metadata.
        Returned dictionary contains the following fields:
        - expert_list
        - rank
        - lora_alpha
        - scaling
        - layer:
            contains layer state_dict and expert indices
        """
        # If expert_list_path is provided, load experts on the fly
        if expert_list_path:
            print(f"Loading experts on the fly from {expert_list_path}")
            # Load expert list from file
            with open(expert_list_path, "r") as f:
                expert_name = f.read()

            lora_module_list = expert_name.strip().split("\n")
            cache = load_expert_weights(lora_module_list)
            concat_expert, expert_list = concat_experts(cache)

            all_expert_info = {
                "layers": concat_expert,
                "expert_list": expert_list,
                "rank": [cache[exp]["peft_config"].r for exp in expert_list],
                "lora_alpha": [
                    cache[exp]["peft_config"].lora_alpha for exp in expert_list
                ],
                "scaling": [
                    cache[exp]["peft_config"].lora_alpha / cache[exp]["peft_config"].r
                    for exp in expert_list
                ],
                "target_modules": list(
                    set().union(
                        *[
                            cache[exp]["peft_config"].target_modules
                            for exp in expert_list
                        ]
                    )
                    - set(["lm_head"])
                ),
            }
        else:
            # Load from pre-saved expert_info_dir
            expert_weight_path = os.path.join(expert_info_dir, "expert_info.pt")
            all_expert_info = torch.load(expert_weight_path)

        if verbose:
            print(
                f"Loaded expert info with {len(all_expert_info['expert_list'])} experts."
            )
            print(f"Target modules: {all_expert_info['target_modules']}")
            print(f"Expert ranks: {all_expert_info['rank']}")
            print(f"LoRA alphas: {all_expert_info['lora_alpha']}")
            print(f"Scaling factors: {all_expert_info['scaling']}")

            for layer_name, layer_info in all_expert_info["layers"].items():
                print(
                    f"Layer: {layer_name}, Weight shape: {layer_info['state_dict'].shape}, Expert indices: {layer_info['index']}"
                )

        return all_expert_info

    def _prepare_adapter_config(self, peft_config, model_config):

        if peft_config.target_modules is None:
            if (
                model_config["model_type"]
                not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
            ):
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                    model_config["model_type"]
                ]
            )
        return peft_config

    @staticmethod
    def _check_target_module_exists(peft_config, key):
        return check_target_module_exists(peft_config, key)

    def _create_and_replace(
        self,
        peft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        lora_A = None
        lora_B = None
        lora_A_layer = None
        lora_B_layer = None
        expert_index = None

        current_key = ".layers." + current_key.split(".layers.", 1)[-1]
        if len(self.all_expert_info["layers"]) > 0:
            for k, v in self.all_expert_info["layers"].items():
                if current_key in k:
                    if "lora_A" in k:
                        lora_A_layer = k
                        lora_A = v["state_dict"]
                        expert_index = v["index"]
                    elif "lora_B" in k:
                        lora_B_layer = k
                        lora_B = v["state_dict"]
                if (
                    lora_A is not None
                    and lora_B is not None
                    and expert_index is not None
                ):
                    break
            if lora_A == None or lora_B == None:
                print(f"{current_key} not found!")
                return
                # raise ValueError(f"No lora weight found for {target}") # jje: ignore cases where lora_A and lora_B does not exist

            del self.all_expert_info["layers"][lora_A_layer]
            del self.all_expert_info["layers"][lora_B_layer]

        adapter_info = {"lora_A": lora_A, "lora_B": lora_B, "index": expert_index}
        new_module = self._create_new_module(
            peft_config=peft_config, target=target, adapter_info=adapter_info
        )
        self._replace_module(parent, target_name, new_module)

    @staticmethod
    def _replace_module(parent_module, child_name, new_module):
        setattr(parent_module, child_name, new_module)

    @staticmethod
    def _create_new_module(
        peft_config: LorahubConfig, target: nn.Module, adapter_info: dict
    ):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, nn.Linear):
            return LorahubLinear(target_base_layer, peft_config, adapter_info)
        else:
            raise NotImplementedError(
                f"Unsupported module type: {type(target_base_layer)}"
            )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        def _set_requires_grad(module, requires_grad):
            for _, p in module.named_parameters():
                p.requires_grad = requires_grad

        _set_requires_grad(self.model, False)
        for module_name, module in self.named_modules():
            if isinstance(module, (LorahubModule)):
                for name, p in module.named_parameters():
                    if "poly_shared_expert_weights" in name:
                        p.requires_grad = True
                    if self.train_lora_weights and "lora_A" in name:
                        p.requires_grad = True
                    if self.train_lora_weights and "lora_B" in name:
                        p.requires_grad = True

    @property
    def device(self):
        return self.model.device

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def set_adapter_layers(self, enabled=True):
        pass

    def set_adapter(self, adapter_name):
        pass
