import os
import torch
import copy
import argparse
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM
from peft import PeftModel
from peft.utils.save_and_load import get_peft_model_state_dict


def main(args):

    base_model_name = args.base_model_name
    expert_list_path = args.expert_list_path
    save_path = args.save_path

    # Load expert list from file
    with open(expert_list_path, "r") as f:
        expert_name = f.read()

    # sort by task and oracle accuracy
    lora_module_list = expert_name.strip().split("\n")
    cache = load_expert_weights(base_model_name, lora_module_list)
    concat_expert, expert_list = concat_experts(cache)

    all_expert_info = {
        "layers": concat_expert,
        "expert_list": expert_list,
        "rank": [cache[exp]["peft_config"].r for exp in expert_list],
        "lora_alpha": [cache[exp]["peft_config"].lora_alpha for exp in expert_list],
        "scaling": [
            cache[exp]["peft_config"].lora_alpha / cache[exp]["peft_config"].r
            for exp in expert_list
        ],
        "target_modules": list(
            set().union(
                *[cache[exp]["peft_config"].target_modules for exp in expert_list]
            )
        ),
    }

    os.makedirs(save_path, exist_ok=True)
    torch.save(all_expert_info, os.path.join(save_path, "expert_info.pt"))


def load_expert_weights(base_model_name, lora_module_list):
    """Load experts from huggingface and save the state_dict and peft_config"""

    cache = {}
    for peft_model_id in tqdm(lora_module_list):
        cache[peft_model_id] = {}
        print("> Loading {} ...".format(peft_model_id))
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16
        )
        cur_peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
        cache[peft_model_id]["state_dict"] = copy.deepcopy(
            get_peft_model_state_dict(cur_peft_model)
        )
        cache[peft_model_id]["peft_config"] = cur_peft_model.peft_config["default"]
        del base_model

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


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--expert_list_path", type=str, required=True, help="path to expert list file"
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="path to save expert_info.pt"
    )

    return parser


if __name__ == "__main__":
    # print("done")
    parser = build_parser()
    args = parser.parse_args()
    main(args)

"""
python step1_load_experts.py \
    --base_model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --expert_list_path "path/to/expert_list.txt" \
    --save_path "path/to/output/expert_info.pt"
"""
