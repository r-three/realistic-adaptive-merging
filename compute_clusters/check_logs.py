#!/usr/bin/env python3

import os
import argparse
import random
from collections import defaultdict


def read_job_names(job_list):
    """Read job names from file, one per line."""
    with open(job_list, "r") as f:
        return [line.split(":")[0].strip() for line in f if line.strip()]


def get_last_line(log_file) -> str:
    """Get the last non-empty line from a log file."""
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            # Find the last non-empty line
            for line in reversed(lines):
                line = line.strip()
                if line:
                    return line
        return "EMPTY_LOG"
    except FileNotFoundError:
        return "LOG_NOT_FOUND"
    except Exception as e:
        return f"ERROR_READING_LOG: {str(e)}"


def postprocess_last_line(line: str) -> str:
    """Post-process the last line to group similar endings."""
    if "Find logs at:" in line:
        return "wandb: Find logs at: <path>"
    return line


def main():
    parser = argparse.ArgumentParser(
        description="Analyze job log files and bucket by last line"
    )
    parser.add_argument(
        "--job_list",
        "-l",
        default="exps_to_check.txt",
        type=str,
        help="File containing list of job names, one per line",
    )
    parser.add_argument(
        "--num_display",
        "-n",
        type=int,
        default=20,
        help="Number of jobs to display per bucket",
    )
    parser.add_argument(
        "--log_folder", default="logs", type=str, help="Folder containing the log files"
    )

    args = parser.parse_args()

    if not os.path.exists(args.job_list):
        print(f"Error: Job list file '{args.job_list}' not found")
        return 1

    if not os.path.exists(args.log_folder):
        print(f"Error: Log folder '{args.log_folder}' not found")
        return 1

    # Read job names
    job_names = read_job_names(args.job_list)
    print(f"Found {len(job_names)} job names")

    # Bucket jobs by their last line
    buckets = defaultdict(list)

    for job_name in job_names:
        log_file = os.path.join(args.log_folder, f"{job_name}.out")
        last_line = get_last_line(log_file)
        last_line = postprocess_last_line(last_line)
        buckets[last_line].append(job_name)

    # Print results
    print(f"\nFound {len(buckets)} different ending patterns:")
    print("=" * 80)

    for last_line, jobs in sorted(
        buckets.items(), key=lambda x: len(x[1]), reverse=True
    ):
        print(f"\nLast line: {last_line}")
        print(f"Count: {len(jobs)}")

        # Sample up to 5 jobs randomly
        sample_jobs = random.sample(jobs, min(args.num_display, len(jobs)))
        sorted_sample_jobs = sorted(sample_jobs)
        print(f"Sample jobs:")
        print("\n".join(sorted_sample_jobs))
        print("-" * 40)


if __name__ == "__main__":
    main()
