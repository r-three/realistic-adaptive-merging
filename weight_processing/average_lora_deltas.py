#!/usr/bin/env python3
"""
Script to compute averaged weight changes from a list of LoRA modules and save them as a checkpoint.

Usage:
    python average_lora_deltas.py --lora_modules_file path/to/lora_modules.txt --output_checkpoint path/to/output.pt --base_model path/to/base/model

Author: Generated script based on average_lora_weights.py
"""

import torch
import os
import argparse
from typing import Dict, List, Optional
import json
from tqdm import tqdm
from huggingface_hub import snapshot_download
import gc
import copy
import shutil
from collections import defaultdict

from transformers import AutoModelForCausalLM
from peft import PeftModel

# Disable progress bars to keep output clean
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def load_lora_modules_list(file_path: str) -> List[str]:
    """
    Load LoRA module paths/IDs from a text file.

    Args:
        file_path: Path to text file containing one LoRA module path/ID per line

    Returns:
        List of LoRA module paths/IDs
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"LoRA modules file not found at {file_path}")

    with open(file_path, "r") as f:
        modules = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Loaded {len(modules)} LoRA module paths/IDs")
    return modules


def download_model_if_needed(model_id: str, cache_dir: str) -> Optional[str]:
    """
    Download model to cache directory if not already present and return local path.

    Args:
        model_id: HuggingFace model ID
        cache_dir: Local cache directory

    Returns:
        Local path to model or None if download failed
    """
    local_dir = os.path.join(cache_dir, model_id.replace("/", "_"))

    # Check if already exists locally
    if os.path.exists(local_dir):
        return local_dir

    # Download if not present
    try:
        print(f"Downloading {model_id} to cache...")
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
        )
        return downloaded_path
    except Exception as e:
        print(f"Error downloading model {model_id}: {e}")
        return None


def compute_lora_delta(
    base_model, lora_path: str, device: str = "cpu", model_id: str = None
) -> Dict[str, torch.Tensor]:
    """
    Compute weight differences (deltas) between LoRA-merged model and base model.

    Args:
        base_model: Base model to compare against
        lora_path: Path to LoRA weights
        device: Device to use for computation
        model_id: Model ID for error handling and re-download

    Returns:
        Dictionary mapping parameter names to delta tensors
    """
    # Clone base model to avoid modifying original
    base = copy.deepcopy(base_model)
    base.to(device)

    try:
        # Load LoRA model
        model = PeftModel.from_pretrained(base, lora_path)
    except Exception as e:
        print(f"Failed to load LoRA from {lora_path}: {e}")
        if model_id:
            print(f"Attempting to delete and re-download {model_id}")
            # Delete the corrupted folder
            if os.path.exists(lora_path):
                shutil.rmtree(lora_path)
            # Re-download the model
            cache_dir = os.path.dirname(lora_path)
            downloaded_path = download_model_if_needed(model_id, cache_dir)
            if downloaded_path:
                model = PeftModel.from_pretrained(base, downloaded_path)
            else:
                raise Exception(f"Failed to re-download model {model_id}")
        else:
            raise e

    # Merge LoRA weights into base model
    print(f"Merging LoRA weights for {lora_path}")
    model = model.merge_and_unload()
    model.to(device)

    # Compute deltas (differences from base model)
    deltas = {}
    base_state_dict = base_model.state_dict()

    for name, param in model.named_parameters():
        if name in base_state_dict:
            base_param = base_state_dict[name].to(device)
            delta = param.data - base_param.data
            # Only store non-zero deltas to save memory
            if delta.abs().max() > 0.0:
                deltas[name] = delta.clone().cpu()  # Move to CPU to save GPU memory

    # Cleanup
    del model
    del base
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return deltas


def compute_averaged_deltas(
    lora_modules: List[str], base_model, cache_dir: str = None, device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Compute averaged weight changes across multiple LoRA modules.

    Args:
        lora_modules: List of LoRA module paths or HuggingFace model IDs
        base_model: Base model to compare against
        cache_dir: Cache directory for downloading models (if needed)
        device: Device to use for computation

    Returns:
        Dictionary of averaged deltas
    """
    print(f"Computing averaged deltas for {len(lora_modules)} LoRA modules...")

    sum_deltas = {}
    deltas_count = defaultdict(int)
    valid_modules = []

    for idx, module_path in enumerate(
        tqdm(lora_modules, desc="Processing LoRA modules")
    ):
        try:
            # Determine if this is a local path or HuggingFace model ID
            # if os.path.exists(module_path):
            #     # Local path
            #     lora_path = module_path
            #     model_id = None
            # else:
            #     # Assume it's a HuggingFace model ID
            #     model_id = module_path
            #     if cache_dir is None:
            #         raise ValueError(
            #             "cache_dir must be provided for HuggingFace model IDs"
            #         )
            #     lora_path = download_model_if_needed(model_id, cache_dir)
            #     if lora_path is None:
            #         print(f"Skipping {module_path} - failed to download")
            #         continue

            print(f"Processing LoRA module {idx+1}/{len(lora_modules)}: {module_path}")
            deltas = compute_lora_delta(base_model, module_path, device, None)

            # Accumulate deltas
            for name, delta in deltas.items():
                if name not in sum_deltas:
                    sum_deltas[name] = delta.clone()
                else:
                    sum_deltas[name] += delta
                deltas_count[name] += 1

            valid_modules.append(module_path)
            print(
                f"Successfully processed {len(valid_modules)}/{len(lora_modules)} modules"
            )

        except Exception as e:
            print(f"Error processing LoRA module {module_path}: {e}")
            continue

    if not valid_modules:
        raise ValueError("No valid LoRA modules were processed!")

    # Compute averages
    print(f"Computing averaged deltas from {len(valid_modules)} valid modules...")
    averaged_deltas = {}
    for name, sum_delta in sum_deltas.items():
        averaged_deltas[name] = sum_delta / deltas_count[name]

    print(f"Computed averaged deltas for {len(averaged_deltas)} parameters")
    return averaged_deltas


def save_checkpoint(deltas: Dict[str, torch.Tensor], output_path: str):
    """
    Save averaged weight deltas to checkpoint file.

    Args:
        deltas: Dictionary of parameter name -> delta tensor
        output_path: Path to save checkpoint
    """
    print(f"Saving checkpoint to {output_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save deltas directly as dict (same format as original script)
    deltas_cpu = {key: value.cpu() for key, value in deltas.items()}
    torch.save(deltas_cpu, output_path)
    print(f"Successfully saved checkpoint with {len(deltas)} parameter deltas")


def main():
    parser = argparse.ArgumentParser(
        description="Compute averaged LoRA weight deltas and save as checkpoint"
    )
    parser.add_argument(
        "--lora_modules_file",
        type=str,
        required=True,
        help="Path to text file containing LoRA module paths/IDs (one per line)",
    )
    parser.add_argument(
        "--output_checkpoint",
        type=str,
        required=True,
        help="Path to save output checkpoint",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path or HuggingFace ID of base model",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for downloading HuggingFace models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for computation",
    )

    args = parser.parse_args()

    # Load LoRA modules list
    lora_modules = load_lora_modules_list(args.lora_modules_file)

    # Set up cache directory if needed
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        print(f"Using cache directory: {args.cache_dir}")

    # Load base model
    print(f"Loading base model from {args.base_model}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
        print(f"Successfully loaded base model")
    except Exception as e:
        print(f"Error loading base model: {e}")
        return 1

    # Compute averaged deltas
    try:
        averaged_deltas = compute_averaged_deltas(
            lora_modules=lora_modules,
            base_model=base_model,
            cache_dir=args.cache_dir,
            device=args.device,
        )
    except Exception as e:
        print(f"Error computing averaged deltas: {e}")
        return 1

    # Save checkpoint
    try:
        save_checkpoint(averaged_deltas, args.output_checkpoint)
        print("Successfully completed!")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
