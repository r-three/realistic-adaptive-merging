from glob import glob
import argparse
from collections import Counter, defaultdict
import os
import json
import copy
import csv
import re

import logging
from colorama import Fore, Style


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.WHITE,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
        "RESULT": Fore.GREEN + Style.BRIGHT,
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, Fore.WHITE)
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        record.msg = f"{log_color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)


RESULT_LEVEL = 35
logging.addLevelName(RESULT_LEVEL, "RESULT")


segment_break = "\n" * 3 + "=" * 80 + "\n"


def find_common_run_name(run_names: list[str]) -> str:
    run_name_segments = [run_name.split("_") for run_name in run_names]
    flattened_segments = [
        segment for sublist in run_name_segments for segment in sublist
    ]
    segment_repeats = Counter(flattened_segments)
    frequent_segments = {
        segment for segment, count in segment_repeats.items() if count >= len(run_names)
    }
    new_run_names = [
        re.sub(
            r"(\*_)+\*",
            "*",
            "_".join(
                [
                    segment if segment in frequent_segments else "*"
                    for segment in segments
                ]
            ),
        )
        for segments in run_name_segments
    ]
    new_run_name_repeats = Counter(new_run_names)
    most_frequent_run_name = max(new_run_name_repeats, key=new_run_name_repeats.get)
    return most_frequent_run_name


def main(args):
    # Job start
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)

    # Load variant patterns and match runs
    logger.info(segment_break + "Job start")
    logger.info(f"Args: {args}")

    # Load variant patterns
    import yaml

    with open(args.variant_pattern, "r") as f:
        variant_patterns = yaml.safe_load(f)

    logger.info(f"Loaded {len(variant_patterns)} variant patterns, globbing runs...")
    variant_runs = {}
    for element in variant_patterns:
        variant_name = element["name"]
        pattern = os.path.join(element["re"], "job_summary.json")
        matched_dirs = glob(pattern)
        variant_runs[variant_name] = matched_dirs
        if len(matched_dirs) == 0:
            logger.warning(
                f"  '{variant_name}': No matches found for pattern {pattern}"
            )
        else:
            logger.info(f"  '{variant_name}': {len(matched_dirs)} matches")

    run_outputs = list(set().union(*variant_runs.values()))

    # Load job summaries
    results: dict[str, dict] = {}
    for run_output in run_outputs:
        with open(run_output, "r") as f:
            job_summary = json.load(f)
        run_name = job_summary["run_name"]
        if run_name in results:
            raise ValueError(f"Duplicate run_name found: {run_name}")
        results[run_name] = job_summary

        # Assign variant name from pattern matching
        found = False
        for variant_name, variant_paths in variant_runs.items():
            if run_output in variant_paths:
                found = True
                results[run_name]["__variant_name__"] = variant_name
                break

        if not found:
            raise ValueError(f"Cannot find variant name for {run_name}")

    logger.info(f"Loading complete. Found {len(results)} run summaries.")

    # Helper functions
    def non_overwrite_save(dict, key, value):
        while key in dict:
            logger.warning(f"Key {key} already exists, adding +")
            key = key + "+"
        dict[key] = value

    # def dict_to_str(result_dict: dict, ignore_keys: list[str] = []) -> str:
    #     ordered_key = sorted(list(result_dict.keys()))
    #     return "\n".join(
    #         [
    #             f"{key}: {result_dict[key]}"
    #             for key in ordered_key
    #             if key not in ignore_keys
    #         ]
    #     )

    def group_by(
        results: dict[str, dict], group_by_attrs: list[str]
    ) -> dict[tuple, list[str]]:
        grouped_run_names: dict[tuple[str], list[str]] = {}
        for run_name, job_summary in results.items():
            attr_tuple = tuple(job_summary.get(attr) for attr in group_by_attrs)
            if attr_tuple not in grouped_run_names:
                grouped_run_names[attr_tuple] = []
            grouped_run_names[attr_tuple].append(run_name)
        return grouped_run_names

    # Process random attr
    logger.info(segment_break + "Processing random attr")
    group_by_attrs = ["__variant_name__", args.task_attr, *args.hp_attrs]
    grouped_run_names = group_by(results, group_by_attrs)
    new_results: dict[str, dict] = {}
    skipped = 0
    for attr_tuple, run_names in grouped_run_names.items():
        num_runs = len(run_names)
        if num_runs == 1:
            new_results[run_names[0]] = results[run_names[0]]
            skipped += 1
        else:
            new_run_name = find_common_run_name(run_names)
            new_run_summary = copy.deepcopy(results[run_names[0]])
            new_run_summary["run_name"] = new_run_name
            for metric_attr in args.metric_attrs:
                values = [results[run_name][metric_attr] for run_name in run_names]
                new_run_summary[metric_attr] = sum(values) / num_runs
            new_run_summary[args.random_attr] = None
            non_overwrite_save(new_results, new_run_name, new_run_summary)
            logger.info(
                f"{num_runs} runs ({run_names}) for {attr_tuple},\naveraging and saving as {new_run_name}."
            )
    if skipped > 0:
        logger.info(f"Skipped {skipped} single-run groups")
    results = new_results

    # Process hp attrs
    logger.info(segment_break + "Processing hp attrs")
    group_by_attrs = ["__variant_name__", args.task_attr]
    grouped_run_names = group_by(results, group_by_attrs)
    new_results: dict[str, dict] = {}
    skipped = 0
    for attr_tuple, run_names in grouped_run_names.items():
        num_runs = len(run_names)
        if num_runs == 1:
            new_results[run_names[0]] = results[run_names[0]]
            skipped += 1
        else:
            new_run_name = find_common_run_name(run_names)
            selected_run_name = None
            best_value = None
            for run_name in run_names:
                if (
                    best_value is None
                    or (
                        args.hp_select_type == "max"
                        and results[run_name][args.hp_select_attr] > best_value
                    )
                    or (
                        args.hp_select_type == "min"
                        and results[run_name][args.hp_select_attr] < best_value
                    )
                ):
                    best_value = results[run_name][args.hp_select_attr]
                    selected_run_name = run_name
            if selected_run_name is None:
                raise ValueError(f"No valid run found for {attr_tuple}")
            new_job_summary = copy.deepcopy(results[selected_run_name])
            new_job_summary["run_name"] = new_run_name
            non_overwrite_save(new_results, new_run_name, new_job_summary)
            logger.info(
                f"{num_runs} runs ({run_names}) for {attr_tuple},\nselecting {selected_run_name} based on {args.hp_select_attr} = {best_value} and saving as {new_run_name}."
            )
    if skipped > 0:
        logger.info(f"Skipped {skipped} single-run groups")
    results = new_results

    # Process task attrs
    logger.info(segment_break + "Parsing tasks")
    # Preserve order from vares.yaml instead of sorting
    variant_names = [element["name"] for element in variant_patterns]
    # Filter to only variants that have results
    variant_names = [
        vn
        for vn in variant_names
        if any(results[run_name]["__variant_name__"] == vn for run_name in results)
    ]
    task_names = set(results[run_name][args.task_attr] for run_name in results)
    tasks_filter = os.environ.get("TASKS")
    if tasks_filter:
        allowed_tasks = set(tasks_filter.split())
        task_names = task_names & allowed_tasks
        logger.info(f"Filtered to {len(task_names)} tasks from TASKS env var")
    task_names = sorted(list(task_names))
    variant_results = {
        variant_name: {task_name: None for task_name in task_names}
        for variant_name in variant_names
    }
    for run_name, job_summary in results.items():
        variant_name = job_summary["__variant_name__"]
        task_name = job_summary[args.task_attr]
        metric_value = None
        for metric_attr in args.metric_attrs:
            metric_value = job_summary.get(metric_attr, metric_value)
        if metric_value is None:
            logger.warning(
                f"Warning: {metric_attr} not found in {run_name} ({variant_name}, {task_name})"
            )
        variant_results[variant_name][task_name] = metric_value

    # Save csv results
    csv_rows = []
    csv_rows.append(["task \\ variant"] + variant_names)
    for task_name in task_names:
        row = [task_name]
        for variant_name in variant_names:
            metric_value = variant_results[variant_name][task_name]
            if metric_value is None:
                row.append("")
            else:
                row.append(f"{metric_value:.3f}")
        csv_rows.append(row)
    with open(args.output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    output_realpath = os.path.realpath(args.output_file)
    logger.info(segment_break + f"Saved results to {output_realpath}")
    logger.warning(f"collect result by \n scp killarney:{output_realpath} ~/Desktop/")

    # Convert to pastable strings and print to logger
    logger.info(f"Results")
    # Print task names
    task_names_str = ",," + ",".join(task_names)
    logger.log(RESULT_LEVEL, task_names_str)
    for idx, variant in enumerate(variant_names):
        column = [csv_rows[i][idx + 1] for i in range(len(csv_rows))]
        variant_result_str = "," + ",".join(column)
        logger.log(RESULT_LEVEL, variant_result_str)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant_pattern",
        type=str,
        default="vares.yaml",
        help="YAML file with variant_name: regex_pattern pairs for matching run directories",
    )
    parser.add_argument("--task_attr", nargs="*", default="task")
    parser.add_argument(
        "--hp_attrs",
        nargs="*",
        default=["lr", "batch_size", "effective_batch_size", "step", "epoch"],
        help="experiment with different hyperparameters will be aggregated according to metric_type",
    )
    parser.add_argument(
        "--random_attr",
        type=str,
        default="seed",
        help="experiment with different random seeds will be averaged (before hp_attrs max/min)",
    )
    parser.add_argument(
        "--metric_attrs",
        nargs="+",
        default=["test/exact_string_match_accuracy"],
    )
    parser.add_argument(
        "--hp_select_attr", type=str, default="exact_string_match_accuracy"
    )
    parser.add_argument("--hp_select_type", type=str, default="max")
    parser.add_argument("--output_file", type=str, default="results.csv")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)


"""
python scripts/results_to_csv.py 
"""
