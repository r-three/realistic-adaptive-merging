"""
Hub Expert Selection Script

This script selects relevant hub experts for a single target task using different methods:
- evaluation: Performance-based selection using task evaluation scores
- closeness: Weight similarity-based selection using LoRA weight comparisons
- proxy: MOOSE model importance-based selection using routing weights
- random: Random selection for baseline comparisons
- rank: LoRA rank-based selection using PEFT configurations

Input: model_ids.txt file (list of hub experts), target_model_path and some method-specific information.
Output: selected experts as a list of model ids saved to output file
"""

import argparse
import os
import math
import random
import json
import glob
import pandas as pd
from tqdm import tqdm
from typing import List, Literal, Dict

from utils.misc_utils import to_comma_name
from weight_processing.collect_lora_experts import (
    get_normalized_weights,
    get_peft_config,
)
from scripts.model_closeness_metrics import l2_closeness, cosine_closeness


def load_cache(cache_dir: str, cache_filename: str) -> Dict[str, float]:
    """
    Load cached scores from JSON file in cache directory.

    Args:
        cache_dir: Path to cache directory
        cache_filename: Filename for this specific cache

    Returns:
        Dictionary of cached scores, or empty dict if cache miss
    """
    if not cache_dir:
        return {}

    cache_file_path = os.path.join(cache_dir, cache_filename)
    if not os.path.exists(cache_file_path):
        return {}

    try:
        with open(cache_file_path, "r") as f:
            cached_data = json.load(f)
        print(f"Using cached scores from {cache_file_path}")
        return cached_data.get("scores", {})
    except Exception as e:
        print(f"Error loading cache: {e}")

    return {}


def save_cache(
    cache_dir: str, cache_filename: str, scores: Dict[str, float], metadata: Dict = None
) -> None:
    """
    Save scores to JSON cache file in cache directory.

    Args:
        cache_dir: Path to cache directory
        cache_filename: Filename for this specific cache
        scores: Dictionary of scores to cache
        metadata: Additional metadata to store
    """
    if not cache_dir:
        return

    try:
        os.makedirs(cache_dir, exist_ok=True)
        cache_data = {"scores": scores}
        if metadata:
            cache_data.update(metadata)

        cache_file_path = os.path.join(cache_dir, cache_filename)
        with open(cache_file_path, "w") as f:
            json.dump(cache_data, f, indent=2)
        print(f"Saved scores cache to {cache_file_path}")
    except Exception as e:
        print(f"Error saving cache: {e}")


def load_evaluation_results_df(
    evaluation_csv_path: str, job_summary_dir: str | None = None
) -> pd.DataFrame:
    """
    Load evaluation results from CSV cache or build from job summary files.

    Args:
        evaluation_csv_path: Path to the evaluation results CSV file
        job_summary_dir: Path to job summary folder (optional, for building cache)

    Returns:
        DataFrame with columns: task, model_name, accuracy
    """
    print(f"Loading evaluation results from {evaluation_csv_path}")

    # Try to load from CSV cache first
    if os.path.exists(evaluation_csv_path):
        print(f"Found cached CSV file: {evaluation_csv_path}")
        df = pd.read_csv(evaluation_csv_path)
        print(f"Loaded {len(df)} rows from CSV cache")
        return df

    # If CSV doesn't exist, build from job_summary_dir
    if not job_summary_dir:
        raise FileNotFoundError(
            f"CSV not found at {evaluation_csv_path} and no job_summary_dir provided to build it"
        )

    if not os.path.exists(job_summary_dir):
        raise FileNotFoundError(f"Job summary directory not found: {job_summary_dir}")

    print(f"Building evaluation results from job summaries in {job_summary_dir}")

    # Find all job_summary.json files in subdirectories
    job_summary_pattern = os.path.join(job_summary_dir, "*", "job_summary.json")
    job_summary_files = glob.glob(job_summary_pattern)
    print(f"Found {len(job_summary_files)} job summary files")

    # Collect all results from JSON files
    results = []

    for job_file in tqdm(job_summary_files, desc="Loading job summaries"):
        try:
            with open(job_file, "r") as f:
                job_data = json.load(f)

            task = job_data.get("task")
            model_name = job_data.get("model_name")
            accuracy = job_data.get("train+valid/exact_string_match_accuracy")

            if all(x is not None for x in [task, model_name, accuracy]):
                results.append(
                    {
                        "task": task,
                        "model_name": model_name,
                        "accuracy": float(accuracy),
                    }
                )

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"Error reading {job_file}: {e}")
            continue

    # Create DataFrame in long format
    df_long = pd.DataFrame(results)
    print(f"Built DataFrame with {len(df_long)} evaluation results from JSON files")

    # Remove duplicates, keeping the last (most recent) result for each model-task pair
    df_long = df_long.drop_duplicates(subset=["model_name", "task"], keep="last")
    print(f"After deduplication: {len(df_long)} unique evaluation results")

    # Pivot to wide format: model_name as rows, tasks as columns
    df = df_long.pivot(index="model_name", columns="task", values="accuracy")
    df = df.reset_index()  # Make model_name a regular column

    # Rename model_name to Model ID
    df = df.rename(columns={"model_name": "Model ID"})

    # Calculate mean across all task columns (excluding Model ID)
    task_columns = [col for col in df.columns if col != "Model ID"]
    df["Mean"] = df[task_columns].mean(axis=1)

    # Add idx column at the front (remove if already exists)
    if "idx" in df.columns:
        df = df.drop(columns=["idx"])
    df.insert(0, "idx", range(len(df)))

    # Reorder columns: idx, Model ID, Mean, then all task columns
    cols = ["idx", "Model ID", "Mean"] + task_columns
    df = df[cols]

    # Report statistics
    num_models = len(df)
    num_tasks = len(task_columns)
    total_elements = num_models * num_tasks
    num_missing = df[task_columns].isna().sum().sum()
    missing_rate = num_missing / total_elements if total_elements > 0 else 0
    print(f"Evaluation matrix: {num_models} models × {num_tasks} tasks = {total_elements} elements")
    print(f"Missing values: {num_missing} ({missing_rate:.2%})")

    # Save to CSV cache
    csv_dir = os.path.dirname(evaluation_csv_path)
    if csv_dir:  # Only create directory if path has a directory component
        os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(evaluation_csv_path, index=False)
    print(f"Saved cache to {evaluation_csv_path}")

    return df


def get_evaluation_scores(
    target_task: str,
    available_model_ids: List[str],
    evaluation_csv_path: str,
    job_summary_dir: str | None = None,
) -> Dict[str, float]:
    """
    Get evaluation scores for available models on target task.

    Args:
        target_task: Task name
        available_model_ids: List of available model IDs
        evaluation_csv_path: Path to evaluation results CSV file
        job_summary_dir: Path to job summary folder (optional, for rebuilding cache)

    Returns:
        Dictionary mapping model_id to score
    """
    print(f"Getting evaluation scores for task: {target_task}")

    # Load DataFrame (wide format: Model ID as rows, tasks as columns)
    df = load_evaluation_results_df(evaluation_csv_path, job_summary_dir)

    # Check if target task column exists
    if target_task not in df.columns:
        print(f"Warning: Task {target_task} not found in evaluation results")
        return {}

    # Filter for available models and get scores for target task
    task_results = df[df["Model ID"].isin(available_model_ids)][
        ["Model ID", target_task]
    ]

    # Drop rows with NaN values for the target task
    task_results = task_results.dropna(subset=[target_task])

    if task_results.empty:
        print(f"Warning: No evaluation scores found for task {target_task}")
        return {}

    # Convert to dictionary
    model_scores = dict(zip(task_results["Model ID"], task_results[target_task]))

    return model_scores


def get_closeness_scores(
    target_model_path: str,
    available_model_ids: List[str],
    closeness_metric: str = "cosine",
    cosine_variant: Literal[
        "cosine", "clamp", "clamp_pm", "abs", "quasi_fim"
    ] = "cosine",
    cosine_aggregate: Literal["micro", "macro"] = "micro",
    cache_path: str = None,
) -> Dict[str, float]:
    """
    Get closeness scores between target model and available models.

    Args:
        target_model_path: Path to the target model
        available_model_ids: List of available model IDs
        closeness_metric: Closeness metric to use
        cosine_variant: Cosine closeness variant
        cosine_aggregate: Cosine closeness aggregation method
        cache_path: Path to cache directory for results (optional)

    Returns:
        Dictionary mapping model_id to closeness_score
    """
    # Create cache filename
    target_model_name = target_model_path.replace("/","---") #os.path.basename(target_model_path).replace("/", "_")
    cache_filename = f"closeness_{target_model_name}_{closeness_metric}_{cosine_variant}_{cosine_aggregate}_{hash(tuple(sorted(available_model_ids)))}.json"

    # Check cache first
    cached_scores = load_cache(cache_path, cache_filename)
    if cached_scores:
        return cached_scores

    print(f"Loading target model weights from {target_model_path}")
    target_weights = get_normalized_weights(
        target_model_path, apply_alpha=True, raise_error=True
    )
    assert target_weights, "Failed to load target model weights"

    # Compute closeness scores
    closeness_scores = {}
    for model_id in tqdm(available_model_ids, desc="Computing closeness"):
        candidate_weights = get_normalized_weights(
            model_id, apply_alpha=True, raise_error=False
        )
        if not candidate_weights:
            print(f"Skipping {model_id}, failed to load weights")
            continue
        if closeness_metric == "cosine":
            closeness = cosine_closeness(
                candidate_weights,
                target_weights,
                variant=cosine_variant,
                aggregate=cosine_aggregate,
            )
        elif closeness_metric == "l2":
            closeness = l2_closeness(candidate_weights, target_weights)
        else:
            raise ValueError(f"Unsupported closeness metric: {closeness_metric}")
        closeness_scores[model_id] = closeness
        print(f"Processed {model_id}: closeness = {closeness:.4f}")

    # Save to cache
    metadata = {
        "target_model_path": target_model_path,
        "closeness_metric": closeness_metric,
        "cosine_variant": cosine_variant,
        "cosine_aggregate": cosine_aggregate,
        "available_model_ids": available_model_ids,
    }
    save_cache(cache_path, cache_filename, closeness_scores, metadata)

    return closeness_scores


def get_rank_scores(
    available_model_ids: List[str], cache_path: str = None
) -> Dict[str, float]:
    """
    Get rank-based scores for available models using PEFT config.

    Args:
        available_model_ids: List of available model IDs
        cache_path: Path to cache directory for results (optional)

    Returns:
        Dictionary mapping model_id to rank_score
    """
    # Create cache filename
    cache_filename = f"rank_{hash(tuple(sorted(available_model_ids)))}.json"

    # Check cache first
    cached_scores = load_cache(cache_path, cache_filename)
    if cached_scores:
        return cached_scores

    print("Getting PEFT config ranks for model selection")

    # Compute rank scores
    rank_scores = {}
    for model_id in tqdm(available_model_ids, desc="Getting PEFT ranks"):
        peft_config = get_peft_config(model_id)
        if peft_config is None:
            print(f"Skipping {model_id}, failed to load PEFT config")
            continue

        # Get rank from PEFT config
        rank = getattr(peft_config, "r", 0)
        rank_scores[model_id] = float(rank)
        print(f"Processed {model_id}: rank = {rank}")

    # Save to cache
    metadata = {"available_model_ids": available_model_ids}
    save_cache(cache_path, cache_filename, rank_scores, metadata)

    return rank_scores


def get_random_scores(
    available_model_ids: List[str], cache_path: str = None, seed: int = 42
) -> Dict[str, float]:
    """
    Get random scores for available models.

    Args:
        available_model_ids: List of available model IDs
        cache_path: Path to cache directory for results (optional)
        seed: Random seed for reproducible results

    Returns:
        Dictionary mapping model_id to random_score
    """
    # Create cache filename
    cache_filename = f"random_{seed}_{hash(tuple(sorted(available_model_ids)))}.json"

    # Check cache first
    cached_scores = load_cache(cache_path, cache_filename)
    if cached_scores:
        return cached_scores

    print("Generating random scores for model selection")

    # Set seed for reproducible results
    random.seed(seed)

    # Generate random scores
    random_scores = {model_id: random.random() for model_id in available_model_ids}

    # Save to cache
    metadata = {"seed": seed, "available_model_ids": available_model_ids}
    save_cache(cache_path, cache_filename, random_scores, metadata)

    print(f"Generated random scores for {len(random_scores)} models")

    return random_scores


def get_proxy_model_scores(
    proxy_model_path: str,
    available_model_ids: List[str],
    nonlinearity: str = "none",
    cache_path: str = None,
) -> Dict[str, float]:
    """
    Get model importance scores using a proxy MOOSE model.

    Args:
        proxy_model_path: Path to the proxy MOOSE model
        available_model_ids: List of available model IDs to filter by
        nonlinearity: Nonlinearity to apply to importance scores
        cache_path: Path to cache directory for results (optional)

    Returns:
        Dictionary mapping model_id to importance_score
    """
    # Create cache filename
    proxy_model_name = os.path.basename(proxy_model_path).replace("/", "_")
    cache_filename = f"proxy_{proxy_model_name}_{nonlinearity}_{hash(tuple(sorted(available_model_ids)))}.json"

    # Check cache first
    cached_scores = load_cache(cache_path, cache_filename)
    if cached_scores:
        return cached_scores

    from peft import AutoPeftModel

    print(f"Loading proxy MOOSE model from {proxy_model_path}")

    # Load the MOOSE model
    proxy_model = AutoPeftModel.from_pretrained(
        proxy_model_path, trust_remote_code=False
    )

    # Get model importance scores
    importance_scores = proxy_model.get_model_importance_scores(
        nonlinearity=nonlinearity
    )

    # Filter by available model IDs - only include if exists in proxy model
    filtered_scores = {
        model_id: importance_scores[model_id]
        for model_id in available_model_ids
        if model_id in importance_scores
    }

    # Save to cache
    metadata = {
        "proxy_model_path": proxy_model_path,
        "nonlinearity": nonlinearity,
        "available_model_ids": available_model_ids,
    }
    save_cache(cache_path, cache_filename, filtered_scores, metadata)

    return filtered_scores


def main(args: argparse.Namespace) -> None:
    """Main function to run hub expert selection for a single task."""
    print(f"Running hub expert selection with args: {args}")

    # Load available model IDs
    if not os.path.exists(args.model_ids_file):
        raise FileNotFoundError(f"Model IDs file not found: {args.model_ids_file}")

    with open(args.model_ids_file, "r") as f:
        available_model_ids = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(available_model_ids)} candidate models")

    # Get scores based on chosen method
    if args.selection_method == "evaluation":
        # Evaluation-based selection using model performance
        scores = get_evaluation_scores(
            target_task=args.task,
            available_model_ids=available_model_ids,
            evaluation_csv_path=args.evaluation_csv_path,
            job_summary_dir=args.job_summary_dir,
        )
    elif args.selection_method == "closeness":
        # Closeness-based selection using model similarity
        if not args.target_model_path:
            raise ValueError("target_model_path required for closeness-based selection")
        scores = get_closeness_scores(
            target_model_path=args.target_model_path,
            available_model_ids=available_model_ids,
            closeness_metric=args.closeness_metric,
            cosine_variant=args.cosine_variant,
            cosine_aggregate=args.cosine_aggregate,
            cache_path=args.cache_path,
        )
    elif args.selection_method == "proxy":
        # Proxy-based selection using MOOSE model importance scores
        if not args.proxy_model_path:
            raise ValueError("proxy_model_path required for proxy-based selection")
        scores = get_proxy_model_scores(
            proxy_model_path=args.proxy_model_path,
            available_model_ids=available_model_ids,
            nonlinearity=args.proxy_importance_nonlinearity,
            cache_path=args.cache_path,
        )
    elif args.selection_method == "random":
        # Random selection
        scores = get_random_scores(
            available_model_ids, cache_path=args.cache_path, seed=args.random_seed
        )
    elif args.selection_method == "rank":
        # Rank-based selection using PEFT config
        scores = get_rank_scores(available_model_ids, cache_path=args.cache_path)
    else:
        raise ValueError(f"Unknown selection method: {args.selection_method}")

    # Sort scores by score value (descending - higher is better)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Print top scores
    print(f"Obtained {len(scores)} / {len(available_model_ids)} scores")
    print(
        f"\nTop {min(5, len(sorted_scores))} models by {args.selection_method} score:"
    )
    for i, (model_id, score) in enumerate(sorted_scores[:5]):
        print(f"  {i+1}: {model_id} (score: {score:.4f})")

    # Select experts
    # First, get all candidates sorted by score
    all_candidates = [model_id for model_id, _ in sorted_scores]

    # Handle target model inclusion
    if args.include_target_model:
        if not args.target_model_path:
            raise ValueError("target_model_path required for include_target_model")

        # If target model is already in the list, remove it so we can push it to front
        if args.target_model_path in all_candidates:
            all_candidates.remove(args.target_model_path)

        # Insert target model at the front
        all_candidates.insert(0, args.target_model_path)

    # Slice to get the requested number of experts
    selected_experts = all_candidates[: args.num_selected]

    print(
        f"Selected {len(selected_experts)} experts (Target model included: {args.include_target_model})"
    )

    # Save expert list
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write("\n".join(selected_experts))

    print(f"Selected {len(selected_experts)} experts, saved to {args.output_file}")

    # Print selection
    print("\nFinal expert selection:")
    for idx, expert in enumerate(selected_experts):
        print(f"  {idx}: {expert}")


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Select relevant hub experts for a target task"
    )

    parser.add_argument(
        "--model_ids_file",
        type=str,
        required=True,
        help="Path to file containing available candidate model IDs",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output file for selected expert IDs",
    )
    parser.add_argument(
        "--selection_method",
        type=str,
        choices=["evaluation", "closeness", "proxy", "random", "rank"],
        default="evaluation",
        help="Method for selecting experts",
    )
    parser.add_argument(
        "--num_selected", type=int, default=50, help="Number of experts to select"
    )
    parser.add_argument(
        "--include_target_model",
        action="store_true",
        help="Include target model path at the front of the selected experts list",
    )
    parser.add_argument(
        "--target_model_path",
        type=str,
        default=None,
        help="Path to target model",
    )
    parser.add_argument(
        "--job_summary_dir",
        type=str,
        default=None,
        help="Path to job summary folder containing job_summary.json files (used to build evaluation CSV if it doesn't exist)",
    )
    parser.add_argument(
        "--evaluation_csv_path",
        type=str,
        default="evaluation_results_cache.csv",
        help="Path to evaluation results CSV file",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task name of the target task (e.g., super_glue_copa)",
    )
    parser.add_argument(
        "--proxy_model_path",
        type=str,
        default=None,
        help="Path to proxy model for proxy method",
    )
    parser.add_argument(
        "--proxy_importance_nonlinearity",
        type=str,
        choices=["none", "abs", "softmax"],
        default="none",
        help="Nonlinearity to apply to proxy importance scores",
    )

    parser.add_argument(
        "--closeness_metric",
        type=str,
        choices=["cosine", "l2"],
        default="cosine",
        help="Closeness metric for weight-based selection (higher is better)",
    )
    parser.add_argument(
        "--cosine_variant",
        type=str,
        choices=["cosine", "clamp", "clamp_pm", "abs", "quasi_fim"],
        default="cosine",
        help="Cosine closeness variant for more sophisticated computation",
    )
    parser.add_argument(
        "--cosine_aggregate",
        type=str,
        choices=["micro", "macro"],
        default="micro",
        help="Cosine closeness aggregation method",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default=None,
        help="Path to cache directory for storing computed scores",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible random scores",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
