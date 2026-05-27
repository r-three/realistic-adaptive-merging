import gc
import torch
import pandas as pd
import json
import yaml
import os
import argparse

# from lorahub.algorithm import default_l1_regularization
#from scripts.llama_child import Llama
#from scripts.qwen_child import Qwen

from scripts.base_lorahub import *
from scripts.evaluation import evaluate_model
from scripts.model_builder import build_model
from datasets import concatenate_datasets

def main(args):

    NUM_EXPERTS = args.num_experts
    BUDGET = args.budget
    NUM_TRAIN_EXAMPLE = args.num_train_example
    SAMPLE_METHOD = args.sample_method
    expert_file_path = args.expert_file_path
    task = args.task
    base_model_name=args.model_name

    # Initialize model class, get model and tokenizer
    model_cls, model, tokenizer = build_model(args, peft_config=None)    

    # path to the model list
    with open(f"{expert_file_path}", "r") as f:
        expert_name = f.read()

    gc.collect()
    torch.cuda.empty_cache()
    lora_modules = expert_name.split('\n')
    lora_modules = [i for i in lora_modules if i != '']
    assert NUM_EXPERTS == len(lora_modules), "# experts expected should match lora modules passed"
    include_best_lora = True if len([i for i in lora_modules if 'r-three/' in i or 'shared_space/' in i]) != 0 else False

    print(f"Main loras: {lora_modules}")

    # Load data
    data = model_cls.get_data(
        tokenizer=tokenizer,
        max_seq_len=2048,
        filter_long_seq=False,
        data_size=NUM_TRAIN_EXAMPLE,
        combine_train_valid=args.combine_train_valid
    )
    # jje: all data is used to compute the loss and optimize
    if not args.combine_train_valid:
        data["train+valid"] = concatenate_datasets([data["train"], data["valid"]])

    # Start learning
    recommendation, final_model, cache_stats = lorahub_learning(
        base_model_name=base_model_name,
        lora_module_list=lora_modules,
        number_of_loras=NUM_EXPERTS,
        data=data,
        max_inference_step=BUDGET,
        batch_size=1,
        get_loss=custom_get_loss,
        get_regular=default_l1_regularization,
        tokenizer=tokenizer,
        assistant_start_token=model_cls.assistant_start_token,
    )

    fin_lora_used = list(cache_stats.keys())
    print(recommendation)
    print(cache_stats)
    print(f"Lora used: {fin_lora_used}")

    # Do inference
    tokenizer.padding_side = "left"
    # parser to call evaluate_model() function
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--calculate_all_metrics", action="store_true")
    eval_args = parser.parse_args([])

    valid_output_file = f"{args.file_save_dir}/{task}_{NUM_EXPERTS}_expert_{BUDGET}_iter_{NUM_TRAIN_EXAMPLE}_ex_{include_best_lora}bestexp_{SAMPLE_METHOD}_pred_{args.eval_split}.csv"
    generation_results_train, metrics_train = evaluate_model(
        args=eval_args,
        model_cls=model_cls,
        model=final_model,
        tokenizer=tokenizer,
        data=data,
        split=args.eval_split,
        output_file=valid_output_file,
    )
    valid_acc = metrics_train[f"{args.eval_split}/exact_string_match_accuracy"]

    test_output_file = f"{args.file_save_dir}/{task}_{NUM_EXPERTS}_expert_{BUDGET}_iter_{NUM_TRAIN_EXAMPLE}_ex_{include_best_lora}bestexp_{SAMPLE_METHOD}_pred_{args.test_split}.csv"
    generation_results_test, metrics_test = evaluate_model(
        args=eval_args,
        model_cls=model_cls,
        model=final_model,
        tokenizer=tokenizer,
        data=data,
        split=args.test_split,
        output_file=test_output_file,
    )
    test_acc = metrics_test[f"{args.test_split}/exact_string_match_accuracy"]

    # Save the result
    final_res = {
        "task": task,
        "expert_name": fin_lora_used,
        f"{args.eval_split}_accuracy": valid_acc,
        f"{args.test_split}_accuracy": test_acc,
        "weights": recommendation.tolist(),
    }

    print(final_res)
    file_name = f"lorahub_grad_free_{task}_{NUM_EXPERTS}expert_{BUDGET}iter_{NUM_TRAIN_EXAMPLE}datasize_{include_best_lora}bestexp_{SAMPLE_METHOD}_result.json"
    with open(f"{args.file_save_dir}/{file_name}", "w") as f:
        json.dump(final_res, f, indent=4)

    final_model.to("cpu")
    del final_model
    gc.collect()
    torch.cuda.empty_cache()


def default_l1_regularization(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--num_experts",
            type=int,
            help="number of experts to sample")
    parser.add_argument(
            "--budget",
            type=int,
            help="max optimization step")
    parser.add_argument(
            "--num_train_example",
            type=int,
            help="training sample size, typically small")
    parser.add_argument(
            "--sample_method",
            type=str,
            help="expert sample method - one of random or oracle")
    parser.add_argument(
            "--task",
            type=str,
            help="specific task under the dataset")
    parser.add_argument(
            "--expert_file_path",
            type=str,
            help="file containing list of experts")
    parser.add_argument(
            "--model_name",
            type=str,
            default="meta-llama/Llama-3.1-8B-Instruct",
            help="base model name or path")
    parser.add_argument(
            "--use_quantized",
            action="store_true",
            help="whether to use quantized model")
    parser.add_argument(
            "--seed",
            type=int,
            default=123,
            help="random seed")
    parser.add_argument(
            "--eval_split",
            type=str,
            default="valid"
            )
    parser.add_argument(
            "--test_split",
            type=str,
            default="test"
            )
    parser.add_argument(
            "--combine_train_valid",
            action="store_true"
            )
    parser.add_argument(
            "--file_save_dir",
            type=str,
            help="location to save the .json result file"
            )


    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
