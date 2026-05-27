"""
Evaluation Job Script

Runs evaluation on models using the evaluation utilities.
Moved from utils/evaluation.py to follow the jobs/ directory pattern.
"""

import gc
import time
import os
import logging
import argparse
import json
import torch
import glob
import shutil
from collections import OrderedDict

from scripts.evaluation import evaluate_model
from scripts.model_builder import build_model
from utils.logging_utils import setup_logging


def find_best_model(tuning_output_pattern, logger):
    """
    Find the best model based on validation loss from tuning outputs.

    Args:
        tuning_output_pattern: glob pattern to search for tuning outputs
        logger: logger instance

    Returns:
        tuple: (best_model_path, best_job_summary_path)
    """
    matched_dirs = glob.glob(tuning_output_pattern)
    logger.info(f"Found {len(matched_dirs)} directories matching pattern")

    best_loss = float("inf")
    best_model_path = None
    best_job_summary_path = None

    for dir_path in matched_dirs:
        job_summary_path = os.path.join(dir_path, "job_summary.json")

        if not os.path.exists(job_summary_path):
            raise FileNotFoundError(f"job_summary.json not found in {dir_path}")

        with open(job_summary_path, "r") as f:
            job_summary = json.load(f)

        val_loss = job_summary["best_validation_loss"]
        logger.info(f"{dir_path}: validation loss = {val_loss}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_path = os.path.join(dir_path, "final_model")
            best_job_summary_path = job_summary_path

    logger.info(f"Best model: {best_model_path} with validation loss: {best_loss}")
    return best_model_path, best_job_summary_path


def main(args, logger):
    logger.info("Starting main function")
    logger.info("Args: \n" + str(args))
    job_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(job_dir, exist_ok=True)
    time0 = time.time()

    gc.collect()
    torch.cuda.empty_cache()

    # If tuning_output_pattern is specified, find the best model
    if args.tuning_output_pattern:
        best_model_path, best_job_summary_path = find_best_model(
            args.tuning_output_pattern, logger
        )
        args.model_name = best_model_path

        # Copy final_model directory
        target_model_dir = os.path.join(job_dir, "final_model")
        if os.path.exists(target_model_dir):
            shutil.rmtree(target_model_dir)
        shutil.copytree(best_model_path, target_model_dir)

        # Copy job_summary.json
        target_job_summary = os.path.join(job_dir, "tuning_job_summary.json")
        if os.path.exists(target_job_summary):
            os.remove(target_job_summary)
        shutil.copy2(best_job_summary_path, target_job_summary)

        logger.info(f"Copied best model to {target_model_dir}")
        logger.info(f"Copied job summary to {target_job_summary}")

    torch.cuda.empty_cache()
    model_cls, model, tokenizer = build_model(args)

    # Apply model modifier if provided (used by delta_evaluation)
    if hasattr(args, "model_modifier") and args.model_modifier is not None:
        model = args.model_modifier(model)

    args.split = args.split.split(",")
    data = model_cls.get_data(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        combine_train_valid="train+valid" in args.split,
        data_size=args.data_size,
    )

    output_file = os.path.join(job_dir, "predictions.csv")

    generation_results, metrics = evaluate_model(
        args,
        model_cls,
        model,
        tokenizer,
        data,
        split=args.split,
        output_file=output_file,
    )

    logger.info(f"Evaluation table first five rows: \n{generation_results.head()}")
    metrics_str = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Metrics: \n{metrics_str}")

    job_summary = OrderedDict([(k, v) for k, v in args.__dict__.items()])
    job_summary.update(metrics)

    time1 = time.time()
    logger.info(f"Time taken: {time1 - time0:.2f} seconds")
    job_summary["time_taken"] = (time1 - time0) / 3600  # convert to hours
    with open(os.path.join(job_dir, "job_summary.json"), "w") as f:
        json.dump(job_summary, f)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluation Job")

    # logging and path
    parser.add_argument("--run_name", type=str, default="eval_debug")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/",
        help="Output directory to save results",
    )

    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="model checkpoint from huggingface or path to load the model from",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="meta-llama/Llama-3.1-8B-Instruct",
        type=str,
        help="path to tokenizer if different from model name (e.g. when loading hf experts)",
    )
    parser.add_argument(
        "--tuning_output_pattern",
        type=str,
        default=None,
        help="glob pattern to search for tuning outputs. If specified, will find the best model based on validation loss",
    )

    # data
    parser.add_argument("--task", default=None, type=str, help="evaluation task")
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="max sequence length for tokenizer",
    )
    parser.add_argument("--seed", type=int, default=123, help="rand seed")
    parser.add_argument(
        "--data_size",
        type=int,
        default=100,
        help="data size this model was trained on; used for logging results",
    )

    # evaluation
    parser.add_argument(
        "--eval_batch_size", default=8, type=int, help="batch size for eval"
    )
    parser.add_argument(
        "--split",
        default="train+valid",
        type=str,
        help="split to evaluate the performance",
    )
    parser.add_argument(
        "--calculate_all_metrics",
        action="store_true",
        help="whether to calculate all eval metrics (bleu, rouge, seq accuracy)",
    )

    # model loading options
    parser.add_argument(
        "--use_quantized",
        action="store_true",
        help="whether to load quantized base model",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="whether to use flash attention for inference",
    )
    parser.add_argument(
        "--use_safetensor",
        action="store_true",
        help="whether to load the model in safetensor format",
    )

    return parser


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    parser = build_parser()
    args = parser.parse_args()
    logger = setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args, logger)
