import os
import time
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from collections import OrderedDict
from typing import List, Dict, Literal
from utils.logging_utils import setup_logging
from utils.merge_utils import uniform_aggregate_stream, ties_stream, tsv_merge_stream
from scripts.model_builder import build_model
from scripts.evaluation import evaluate_model


def create_merged_model(
    base_model,
    merged_task_vector: Dict[str, torch.Tensor],
):

    model_keys = list(dict(base_model.named_parameters()).keys())
    with torch.no_grad():
        for name, p in base_model.named_parameters():
            layer = name.replace('model','').replace('.weight','')
            if layer in merged_task_vector:
                print(f"{layer} in merged task vector state dict")
                p.add_(merged_task_vector[layer].to(p.device))

    return base_model


def build_parser():
    parser = argparse.ArgumentParser(description="Merging job")
    
    # logging and path
    parser.add_argument("--run_name", type=str, default="tuning_debug")
    parser.add_argument("--output_dir", type=str, default="outputs/")

    # Other
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--task", type=str)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--data_size", type=int, default=100)
    parser.add_argument("--use_quantized", action="store_true")
    parser.add_argument("--eval_batch_size", default=4, type=int, help="evaluation batch size")
    parser.add_argument("--subset_sample_path", type=str, default="auto")
    parser.add_argument("--eval_split", type=str, default="valid,test", help="which split to use for evaluation, train, valid or test")
    parser.add_argument("--save_expert_task_vector", action="store_true")
    #parser.add_argument("--cache_path", type=str, default="outputs/merged_weights")
    parser.add_argument("--seed", type=int, default=123)

    # merging arg
    parser.add_argument("--merging_method", type=str, choices=["aggregate", "ties", "core_space", "tsv"])
    parser.add_argument("--expert_file_path", type=str, default="")
    parser.add_argument("--expert_aggregate_state_dict_path", type=str, default="")
    parser.add_argument("--expert_ties_state_dict_path", type=str, default="")
    parser.add_argument("--expert_tsv_state_dict_path", type=str, default="")
    parser.add_argument("--aggregate_method", type=str, choices=["sum", "avg", "sign"], default="sum")
    parser.add_argument("--majority_sign_method", type=str, choices=["total", "frequency"], default="total")
    parser.add_argument("--density", type=float, default=1)
    parser.add_argument("--weight_init", type=str, default="equal_int")
    parser.add_argument("--tsv_reduced_size", type=int, default=16)
    parser.add_argument("--num_chunks", type=int, default=16)
    parser.add_argument("--num_base_model_layers", type=int, default=32)
    
    return parser


def main(args, logger):

    job_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(job_dir, exist_ok=True)

    time0 = time.time()
    # Load the model
    model_cls, model, tokenizer = build_model(args, peft_config=None)
    
    # Load the data
    data = model_cls.get_data(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        filter_long_seq=False,
        data_size=args.data_size,
        subset_sample_path=args.subset_sample_path
    )

    # Get the adapters merged
    logger.info("Computing the merged task vector...")
    processed_experts = None
    file_save_path = None
    if args.merging_method == "ties":
        if os.path.isfile(args.expert_ties_state_dict_path):
            merged_task_vector = torch.load(args.expert_ties_state_dict_path)
        else:
            logger.info(f"Creating {args.expert_ties_state_dict_path}...")
            merged_task_vector, processed_experts = ties_stream(
                    expert_file_path=args.expert_file_path,
                    expert_aggregate_state_dict_path=args.expert_aggregate_state_dict_path,
                    weight_init=args.weight_init,
                    density=args.density,
                    majority_sign_method=args.majority_sign_method
            )
            file_save_path = args.expert_ties_state_dict_path
    elif args.merging_method == "aggregate":
        if os.path.isfile(args.expert_aggregate_state_dict_path):
            merged_task_vector = torch.load(args.expert_aggregate_state_dict_path)
        else:
            logger.info(f"Creating {args.expert_aggregate_state_dict_path}...")
            merged_task_vector, processed_experts = uniform_aggregate_stream(
                    expert_file_path=args.expert_file_path,
                    prune_pct=args.density,
                    aggregate_method=args.aggregate_method
            )
            file_save_path = args.expert_aggregate_state_dict_path
    elif args.merging_method == "tsv":
        if os.path.isfile(args.expert_tsv_state_dict_path):
            merged_task_vector = torch.load(args.expert_tsv_state_dict_path)
        else:
            logger.info(f"Creating {args.expert_tsv_state_dict_path}...")
            # calculate layer splits
            chunk_size = args.num_base_model_layers // args.num_chunks
            all_subset_layers=[]
            print(f"Subset layers list: {all_subset_layers}")
            for i in range(args.num_chunks):
                all_subset_layers.append(list(np.arange(i * chunk_size, (i+1) * chunk_size, 1))) 
            
            merged_task_vector = tsv_merge_stream(
                    all_subset_layers=all_subset_layers,
                    expert_file_path=args.expert_file_path,
                    weight_init=args.weight_init,
                    reduced_size=args.tsv_reduced_size
            )
            file_save_path = args.expert_tsv_state_dict_path
            
    else:
        raise ValueError(f"Merging method not implemented: {args.merging_method}")

    if args.save_expert_task_vector and file_save_path is not None:
        if os.path.isfile(file_save_path):
            print(f"{file_save_path} already exists!")
        else:
            logger.info("Saving the computed merged task vector...")
            path = Path(file_save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(merged_task_vector, file_save_path)
            if processed_experts != None:
                with open(f"{file_save_path.replace('.pt','.txt')}", "w") as f:
                    f.write('\n'.join(processed_experts))

        #file_name=f"{args.cache_path}/HF_expert_{args.merging_method}_merged_task_vector"
        #torch.save(merged_task_vector, f"{args.cache_path}/{file_name}.pt")
        #if processed_experts:
        #    with open(f"{args.cache_path}/{file_name}.txt", "w") as f:
        #        f.write('\n'.join(processed_experts))

    # Combine the merged adapters into the base model
    logger.info("Creating a merged model...")
    merged_model = create_merged_model(model, merged_task_vector)
    del merged_task_vector

    # Evaluation
    logger.info("Evaluating...")
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    generation_results, metrics = evaluate_model(
        args,
        model_cls,
        model,
        tokenizer,
        data,
        split=args.eval_split.split(","),
    )
    logger.info(f"Metrics: \n{metrics}")
    time1 = time.time()

    # Save the summary
    job_summary = OrderedDict([(k, v) for k, v in args.__dict__.items()])
    job_summary.update(metrics)
    job_summary["time_taken"] = (time1 - time0) / 3600  # convert to hours
    logger.info(f"Time taken: {time1 - time0:.2f} seconds")

    with open(os.path.join(job_dir, "job_summary.json"), "w") as f:
        json.dump(job_summary, f)
    print("Job summary: \n" + json.dumps(job_summary, indent=4))


if __name__ == '__main__':
    
    parser = build_parser()
    args = parser.parse_args()
    logger = setup_logging()
    main(args, logger)
    
    print("done!")
