#!/usr/bin/env python3
"""
Script to cache repos from tasks.yaml and model list files.
Tracks success/failure in YAML files and skips already successful downloads.
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_dataset
from collections import OrderedDict


def read_datasets_from_tasks_yaml(filepath: str) -> List[Tuple[str, Dict]]:
    """Read dataset configurations from tasks.yaml."""
    with open(filepath, "r") as f:
        tasks = yaml.safe_load(f)

    datasets = []
    seen = set()

    for task_name, task_config in tasks.items():
        # Skip JSON files - they don't need caching
        if "json_file" in task_config:
            continue

        # Create a unique key for this dataset configuration
        if "path" in task_config:
            path_str = str(task_config["path"])
            if path_str not in seen:
                seen.add(path_str)
                datasets.append((task_name, task_config))

    return datasets


def read_models_from_file(filepath: str) -> List[str]:
    """Read model names from a text file (one per line)."""
    with open(filepath, "r") as f:
        models = [line.strip() for line in f if line.strip()]
    return models


def load_status_file(filepath: str) -> OrderedDict:
    """Load status tracking file, returns OrderedDict with failed items first."""
    if not Path(filepath).exists():
        return OrderedDict()

    with open(filepath, "r") as f:
        data = yaml.safe_load(f) or {}

    # Sort so failed items come first
    failed = OrderedDict()
    succeeded = OrderedDict()

    for name, success in data.items():
        if success:
            succeeded[name] = success
        else:
            failed[name] = success

    # Combine with failed first
    result = OrderedDict()
    result.update(failed)
    result.update(succeeded)
    return result


def save_status_file(filepath: str, status: OrderedDict) -> None:
    """Save status tracking file with failed items first."""
    # Re-order to put failed first
    failed = OrderedDict()
    succeeded = OrderedDict()

    for name, success in status.items():
        if success:
            succeeded[name] = success
        else:
            failed[name] = success

    result = OrderedDict()
    result.update(failed)
    result.update(succeeded)

    with open(filepath, "w") as f:
        yaml.dump(dict(result), f, default_flow_style=False, sort_keys=False)


def cache_dataset(task_name: str, task_config: Dict) -> bool:
    """Cache dataset using load_dataset() following base_model.py pattern."""
    try:
        print(f"Loading dataset: {task_name}")

        # Handle combined subsets
        if "combine_subsets" in task_config:
            assert (
                "path" in task_config
            ), "When combining subsets, need to provide the path to the dataset"

            for sub in task_config["combine_subsets"]:
                temp = load_dataset(*task_config["path"], sub)
        # Handle regular datasets
        elif "path" in task_config:
            data = load_dataset(*task_config["path"])
        else:
            raise ValueError("No valid path found in task config")

        print(f"✓ Dataset {task_name} loaded and cached successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to load dataset {task_name}: {e}")
        return False


def cache_model(model_name: str) -> bool:
    """Cache model using hf download."""
    try:
        print(f"Downloading model: {model_name}")

        result = subprocess.run(
            ["hf", "download", model_name],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
        )

        if result.returncode == 0:
            print(f"✓ Model {model_name} downloaded successfully")
            return True
        else:
            print(f"✗ Failed to download model {model_name}: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ Timeout downloading model {model_name}")
        return False
    except Exception as e:
        print(f"✗ Failed to download model {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Cache repo loading with progress tracking")
    parser.add_argument(
        "--tasks_yaml",
        default="tasks.yaml",
        help="Path to tasks.yaml file for datasets",
    )
    parser.add_argument(
        "--models_file",
        default="results/model_lists/refiltered_model_ids.txt",
        help="File containing list of model names",
    )
    parser.add_argument(
        "--dataset_status",
        default="dataset_status.yaml.tmp",
        help="YAML file to track dataset status",
    )
    parser.add_argument(
        "--model_status",
        default="model_status.yaml.tmp",
        help="YAML file to track model status",
    )

    args = parser.parse_args()

    # Validate input files exist
    if not Path(args.tasks_yaml).exists():
        print(f"Error: Tasks file {args.tasks_yaml} does not exist")
        sys.exit(1)

    if not Path(args.models_file).exists():
        print(f"Error: Models file {args.models_file} does not exist")
        sys.exit(1)

    # Read input files
    datasets = read_datasets_from_tasks_yaml(args.tasks_yaml)
    models = read_models_from_file(args.models_file)

    # Load existing status
    dataset_status = load_status_file(args.dataset_status)
    model_status = load_status_file(args.model_status)

    # Add new items to status dicts
    for task_name, _ in datasets:
        if task_name not in dataset_status:
            dataset_status[task_name] = False

    for model in models:
        if model not in model_status:
            model_status[model] = False

    print(f"Caching {len(datasets)} datasets and {len(models)} models")
    print("=" * 60)

    # Cache datasets
    print("Caching Datasets:")
    print("-" * 30)
    dataset_dict = {task_name: config for task_name, config in datasets}
    for task_name in dataset_status.keys():
        if dataset_status[task_name]:
            print(f"⊘ Skipping {task_name} (already successful)")
            continue

        if task_name in dataset_dict:
            success = cache_dataset(task_name, dataset_dict[task_name])
            dataset_status[task_name] = success
            save_status_file(args.dataset_status, dataset_status)

    # Cache models
    print("\nCaching Models:")
    print("-" * 30)
    for model_name in model_status.keys():
        if model_status[model_name]:
            print(f"⊘ Skipping {model_name} (already successful)")
            continue

        success = cache_model(model_name)
        model_status[model_name] = success
        save_status_file(args.model_status, model_status)

    # Summary
    print("\nSummary:")
    print("=" * 60)

    dataset_success = sum(1 for v in dataset_status.values() if v)
    model_success = sum(1 for v in model_status.values() if v)

    print(f"Datasets: {dataset_success}/{len(dataset_status)} successful")
    print(f"Models: {model_success}/{len(model_status)} successful")
    print(f"Total: {dataset_success + model_success}/{len(dataset_status) + len(model_status)} successful")

    print(f"\nStatus files:")
    print(f"  Datasets: {args.dataset_status}")
    print(f"  Models: {args.model_status}")

    dataset_failed = len(dataset_status) - dataset_success
    model_failed = len(model_status) - model_success

    if dataset_failed > 0:
        print(f"\n⚠ {dataset_failed} dataset(s) failed (listed first in {args.dataset_status})")
    if model_failed > 0:
        print(f"⚠ {model_failed} model(s) failed (listed first in {args.model_status})")


if __name__ == "__main__":
    main()
