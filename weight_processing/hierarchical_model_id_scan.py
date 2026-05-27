"""
Script to create model lists hierarchically for LoRA hub experiments.

This script generates model list files that specify which expert models to use
for a given task.
"""

import os
import argparse
import json
import glob
import re


def writelines_to_file(lines: list[str], filepath: str):
    """Write a list of lines to a file, creating directories if needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(lines))


def readlines_from_file(filepath: str):
    """Read lines from a file and return as a list."""
    with open(filepath, "r") as f:
        lines = f.read().strip().split("\n")
    return lines


def readlines_from_multi_file(filepaths: str | list[str]) -> list[str]:
    """Read lines from multiple files and return as a combined list."""
    lines = []
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    for filepath in filepaths:
        lines.extend(readlines_from_file(filepath))
    return lines


def get_model_segments(
    task: str,
    model_list: list[str],
    target_lora_prefix: str,
    output_dir: str,
    segment_length: int | None = None,
    segment_start: int | None = None,
):
    """
    Create a model list file for a given task with a segment of expert models.

    The function loads expert model IDs, takes a segment (slice) of them,
    prepends the target task's LoRA model, and saves to a file.

    Args:
        task: Name of the task (e.g., 'xnli_en', 'arc_easy')
        target_lora_prefix: Prefix path to the target LoRA model (task name will be appended)
        model_id_list: Path(s) to file(s) containing model IDs, or list of paths
        segment_length: Number of expert models to include in the segment
        segment_start: Starting index in the model list for the segment
        output_dir: Directory to save the output model list file
    """
    print(
        f"Processing segment starting at index {segment_start}, length {segment_length}"
    )
    if segment_start is not None:
        output_model_list = [f"{target_lora_prefix}_{task}/final_model"] + model_list[
            segment_start : segment_start + segment_length
        ]

        output_path = os.path.join(
            output_dir,
            task,
            f"len{segment_length}",
            f"start{segment_start}.txt",
        )
    else:
        output_model_list = [f"{target_lora_prefix}_{task}/final_model"] + model_list

        output_path = os.path.join(
            output_dir,
            task,
            f"len{segment_length}",
            f"all_models.txt",
        )
    writelines_to_file(output_model_list, output_path)
    print(f"Created model list at: {output_path}")
    print(f"  Target model: {output_model_list[0]}")
    print(f"  Expert models: {len(output_model_list) - 1}")


def get_all_model_segments(
    task: str,
    model_list: list[str],
    segment_length: int,
    target_lora_prefix: str,
    output_dir="outputs/segment_scan/",
):
    """
    Create model list files for all segments of a task.

    Args:
        task: Task name (e.g., 'xnli_en')
        model_list: List of model IDs/paths already loaded
        segment_length: Size of each segment
        target_lora_prefix: Prefix path to the target LoRA model (task name will be appended)
        output_dir: Directory to save the output files
    """
    num_segments = (len(model_list) + segment_length - 1) // segment_length

    # Check if target folder exists and has the expected number of files
    target_folder = os.path.join(output_dir, task, f"len{segment_length}")
    if os.path.exists(target_folder):
        existing_txt_files = glob.glob(os.path.join(target_folder, "*.txt"))
        if len(existing_txt_files) == num_segments:
            print(
                f"Skipping task {task} - target folder already has {num_segments} .txt files"
            )
            return

    for i in range(num_segments):
        segment_start = i * segment_length
        get_model_segments(
            task=task,
            segment_length=segment_length,
            segment_start=segment_start,
            model_list=model_list,
            target_lora_prefix=target_lora_prefix,
            output_dir=output_dir,
        )


def get_top_model_segments(
    task: str,
    model_list: list[str],
    segment_length: int,
    last_round_segment_length: int,
    num_last_round_segments_to_add: int,
    selection_criteria: str,
    target_lora_prefix: str,
    baseline_lora_prefix: str = "",
    output_dir="outputs/segment_scan/",
    strict: bool = True,
):
    """
    Create model list files for top-performing segments based on selection criteria.

    Args:
        task: Task name
        model_list: List of model IDs/paths already loaded
        segment_length: Size of each segment
        last_round_segment_length: Size of segments in the last round
        num_last_round_segments_to_add: Number of segments to fracture
        selection_criteria: Criteria for selecting top models
        target_lora_prefix: Prefix path to the target LoRA model
        baseline_lora_prefix: Prefix path to the baseline LoRA model
        output_dir: Directory to save the output files
    """
    # 1. Read job_summary.json files from runs
    result_pattern = os.path.join(
        output_dir, task, f"len{last_round_segment_length}", "*", "job_summary.json"
    )
    result_files = glob.glob(result_pattern)

    # Check how many txt files we have at this level
    txt_pattern = os.path.join(
        output_dir, task, f"len{last_round_segment_length}", "*.txt"
    )
    txt_files = glob.glob(txt_pattern)

    if not result_files:
        print(f"Error: No results found at {result_pattern}")
        print("Exiting - run experiments first before fracturing segments")
        return

    if len(result_files) != len(txt_files):
        print(
            f"Warning: Mismatch between results ({len(result_files)}) and txt files ({len(txt_files)})"
        )
        if strict or len(result_files) < len(txt_files) * 0.9:
            print("Exiting - ensure all experiments have completed")
            return
        else:
            print("Continuing in non-strict mode")

    # If baseline_lora_prefix is provided, append result_files
    if baseline_lora_prefix:
        baseline_file = os.path.join(
            f"{baseline_lora_prefix}_{task}", "job_summary.json"
        )
        result_files.append(baseline_file)

    # 2. Rank models based on selection_criteria
    segment_results = []
    for result_file in result_files:
        with open(result_file, "r") as f:
            result = json.load(f)

        # Extract segment_start from folder name using regex
        if baseline_lora_prefix and result_file == baseline_file:
            segment_start = -1  # Baseline segment
        else:
            folder_name = os.path.basename(os.path.dirname(result_file))
            match = re.search(r"start(\d+)", folder_name)
            if not match:
                print(f"Warning: Could not extract segment_start from {folder_name}")
                continue

            segment_start = int(match.group(1))

        # Get metric value based on selection_criteria
        metric_value = result.get(selection_criteria, None)

        if metric_value is not None:
            segment_results.append(
                {
                    "segment_start": segment_start,
                    "metric_value": metric_value,
                    "result_file": result_file,
                }
            )

    if baseline_lora_prefix:
        print(f"Included baseline segment from {baseline_file}")
        print(segment_results[-1])

    if len(segment_results) != len(result_files):
        print(f"Error: Metric '{selection_criteria}' not found in all results")
        print(f"Found metric in {len(segment_results)}/{len(result_files)} files")
        return

    # Sort based on whether we want to minimize or maximize
    minimize_metrics = ["loss", "perplexity", "error"]
    lower_is_better = any(m in selection_criteria.lower() for m in minimize_metrics)
    segment_results.sort(
        key=lambda x: (
            x["metric_value"] if lower_is_better else -x["metric_value"],
            x["segment_start"],
        )
    )

    # 3. Select top num_last_round_segments_to_add segments
    top_segments = segment_results[:num_last_round_segments_to_add]

    print(f"Selected top {len(top_segments)} segments based on {selection_criteria}:")
    for seg in top_segments:
        print(
            f"  Start {seg['segment_start']}: {selection_criteria}={seg['metric_value']:.4f}"
        )

    # 4. Create model list files for these top segments at finer granularity
    if last_round_segment_length > 1:
        # Calculate expected number of subsegments
        expected_subsegments = 0
        for top_seg in top_segments:
            parent_start = top_seg["segment_start"]
            parent_end = parent_start + last_round_segment_length
            for subsegment_start in range(parent_start, parent_end, segment_length):
                if subsegment_start < len(model_list):
                    expected_subsegments += 1

        # Check if target folder exists and has the expected number of files
        target_folder = os.path.join(output_dir, task, f"len{segment_length}")
        if os.path.exists(target_folder):
            existing_txt_files = glob.glob(os.path.join(target_folder, "*.txt"))
            if len(existing_txt_files) == expected_subsegments:
                print(
                    f"Skipping task {task} - target folder already has {expected_subsegments} .txt files"
                )
                return

        # Break down each top segment into smaller segments of size last_round_segment_length
        total_subsegments = 0
        for top_seg in top_segments:
            parent_start = top_seg["segment_start"]
            parent_end = parent_start + last_round_segment_length
            # Create finer segments within this top segment
            for subsegment_start in range(parent_start, parent_end, segment_length):
                if subsegment_start < len(model_list):
                    get_model_segments(
                        task=task,
                        segment_length=segment_length,
                        segment_start=subsegment_start,
                        model_list=model_list,
                        target_lora_prefix=target_lora_prefix,
                        output_dir=output_dir,
                    )
                    total_subsegments += 1

        print(
            f"Generated {total_subsegments} refined segments of size {segment_length} from top {len(top_segments)} segments"
        )
    else:
        # Combine segments
        top_model_list = []
        for idx, top_seg in enumerate(top_segments):
            parent_start = top_seg["segment_start"]
            parent_end = parent_start + last_round_segment_length
            if parent_start == -1:
                print(f"Baseline segment found, stopping combination at index {idx}")
                break
            top_model_list.extend(model_list[parent_start:parent_end])
        print(
            f"Combining top segments into one model list with {len(top_model_list)} models"
        )

        get_model_segments(
            task=task,
            segment_length=segment_length,
            segment_start=0,
            model_list=top_model_list,
            target_lora_prefix=target_lora_prefix,
            output_dir=output_dir,
        )
        print(
            f"Generated combined model list from first {segment_length} models from top segments"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create model lists for LoRA hub experiments with segment scanning"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name (e.g., xnli_en, arc_easy, glue_mrpc)",
    )
    parser.add_argument(
        "--target_lora_prefix",
        type=str,
        default="shared_space/lora_2080/lora_2080_lr3e-4_step400_rank64",
        help="Prefix path to the target LoRA model (task name will be appended)",
    )
    parser.add_argument(
        "--baseline_lora_prefix",
        type=str,
        default="",
        help="Prefix path to the baseline LoRA model (task name will be appended)",
    )
    parser.add_argument(
        "--model_id_file",
        type=str,
        nargs="+",
        default=["results/model_lists/refiltered_model_ids.txt"],
        help="Path(s) to file(s) containing model IDs",
    )
    parser.add_argument(
        "--segment_length", type=int, default=16, help="Size of each segment"
    )
    parser.add_argument(
        "--last_round_segment_length",
        type=int,
        default=None,
        help="Size of last round segment (if specified, enables top model segment mode)",
    )
    parser.add_argument(
        "--num_last_round_segments_to_add",
        type=int,
        default=30,
        help="Number of segments to fracture in top model mode",
    )
    parser.add_argument(
        "--selection_criteria",
        type=str,
        default="valid/exact_string_match_accuracy",
        help="Criteria for selecting top models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/segment_scan/",
        help="Directory to save the output model list file",
    )
    parser.add_argument(
        "--result_directory",
        type=str,
        default="outputs/segment_scan_results/",
        help="Directory containing results for top model selection",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Whether to enforce strict checking of result files",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    model_list = readlines_from_multi_file(args.model_id_file)

    if args.last_round_segment_length is None:
        get_all_model_segments(
            task=args.task,
            model_list=model_list,
            segment_length=args.segment_length,
            target_lora_prefix=args.target_lora_prefix,
            output_dir=args.output_dir,
        )
    else:
        get_top_model_segments(
            task=args.task,
            model_list=model_list,
            segment_length=args.segment_length,
            last_round_segment_length=args.last_round_segment_length,
            num_last_round_segments_to_add=args.num_last_round_segments_to_add,
            selection_criteria=args.selection_criteria,
            target_lora_prefix=args.target_lora_prefix,
            baseline_lora_prefix=args.baseline_lora_prefix,
            output_dir=args.output_dir,
        )
