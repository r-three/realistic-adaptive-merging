import pynvml as nvml
import psutil
import numpy as np
import json
import pandas as pd

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import torch
from functools import partial
from transformers import default_data_collator

def to_comma_name(dataset_name: str, task: str | None) -> str:
    if task is None:
        task = "None"
    return f"{dataset_name.replace('/', ',')},,{task.replace('/', ',')}"


def to_dataset_name(comma_name: str) -> tuple[str, str | None]:
    dataset_name, task = comma_name.split(",,")
    dataset_name = dataset_name.replace(",", "/")
    task = task.replace(",", "/")
    return dataset_name, task if task != "None" else None


def get_mem_usage_stats(msg):
    nvml.nvmlInit()
    for idx in range(nvml.nvmlDeviceGetCount()):
        handle = nvml.nvmlDeviceGetHandleByIndex(idx)
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        total_mem = info.total
        total_mem = total_mem / (1024 * 1024 * 1024)
        free_mem = info.free
        free_mem = free_mem / (1024 * 1024 * 1024)
        print(
            f"State {msg} Device {idx}: total mem {total_mem:.2f}, free mem {free_mem:.2f}"
        )

    print(
        f"State {msg} Total virtual memory available {psutil.virtual_memory().available//1024**3} GB | Virtual memory % used: {np.round(psutil.virtual_memory().percent)}% | CPU memory % used: {np.round(psutil.cpu_percent())}%"
    )


# def compute_llama_instruct_binary_accuracy_metrics(eval_preds):
#     logits, labels = eval_preds

#     # assistant's turn (llama tokenizer)
#     match=list([128006,78191,128007]) # assistant turn cue

#     # 1. extract answer portion from prediction
#     pred_seg_ls=[]
#     logits = logits.argmax(axis=-1)
#     for pred in list(logits):
#         max_idx = len(pred)-1
#         for i in range(max_idx):
#             if list(pred[max_idx-i:max_idx-i+3]) == match:
#                 pred_seg_ls.append(list(pred[max_idx-i+3:max_idx-i+5])) # just take the 2 tokens generated after
#                 break

#     # 2. extract the actual answer
#     ans_ls=[]
#     for ans in list(labels):
#         ans_ls.append(list(ans[-3:-1])) # take the last three except the -100 at the end; masked out

#     # 3. Calculate accuracy
#     correct = 0
#     for i in range(len(ans_ls)):
#         if pred_seg_ls[i] == ans_ls[i]:
#             correct += 1

#     return {"accuracy":correct/len(ans_ls)}


def load_gradient_stats(path):

    with open(path) as f:
        ff = json.load(f)

    ls = []
    for k in ff.keys():
        temp = (
            pd.DataFrame(ff[k]["0"])
            .T.reset_index()
            .rename(columns={"index": "example_num"})
        )
        temp["example_num"] = temp["example_num"].astype(int)
        temp["task"] = k.lower()
        ls.append(temp)
    df_grad = pd.concat(ls)

    return df_grad


def load_proba_stats(path):

    with open(path) as f:
        ff = json.load(f)
    ls = []
    for k in ff.keys():
        temp = (
            pd.DataFrame(ff[k]["0"])
            .T.reset_index()
            .rename(columns={"index": "example_num"})
        )
        temp = temp.explode(
            ["target_proba", "target", "model_pred_proba", "model_pred"]
        )
        temp["example_num"] = temp["example_num"].astype(int)
        temp["task"] = k.lower()
        ls.append(temp)

    df_proba = pd.concat(ls)

    df_proba = (
        df_proba.groupby(["task", "example_num"])
        .agg(
            {
                "target_proba": "mean",
                "model_pred_proba": "mean",
                "avg_error": "mean",
                "avg_confidence": "mean",
                "target": "unique",
                "model_pred": "unique",
            }
        )
        .reset_index()
    )
    return df_proba


def load_auc(path, tasks_to_exclude):

    with open(path) as f:
        ff = json.load(f)

    df_auc = (
        pd.DataFrame(ff)
        .sort_values("extrapolation_auc")
        .rename(columns={"extrapolation_auc": "auc"})
    )
    df_auc = df_auc[~df_auc["task"].isin(tasks_to_exclude)].reset_index(drop=True)

    return df_auc


def load_cosine_stats(path):

    with open(path) as f:
        ff = json.load(f)
        cos = {}
        for k in ff.keys():
            cos[k] = {}
            for iter in ff[k]:
                cos[k][iter] = []
                for b_num in ff[k][iter]:
                    if b_num != "all":
                        tmp = pd.DataFrame.from_dict(
                            ff[k][iter][b_num], orient="index"
                        ).T
                        np.fill_diagonal(tmp.values, None)
                        cos[k][iter].append(tmp)
                t1 = pd.concat(cos[k][iter])
                t2 = pd.concat(cos[k][iter]).T
                org_cols = t1.columns
                t1.index = list(np.arange(0, t1.shape[0], 1))
                t1.columns = list(np.arange(0, t1.shape[0], 1))
                t2.index = list(np.arange(0, t2.shape[0], 1))
                t2.columns = list(np.arange(0, t2.shape[0], 1))
                t1 = t1.fillna(t2)
                t1.columns = org_cols
                t1.index = org_cols
                cos[k][iter] = t1

    d = {}
    ls_df = []
    for k in cos:
        d[k] = []
        for iter in cos[k]:
            d[k].append(cos[k][iter].mean(axis=0).reset_index())
        t = pd.concat(d[k]).rename(columns={"index": "example_num", 0: "cos_sim"})
        t["task"] = k.lower()
        ls_df.append(t)

    df_cos = pd.concat(ls_df)
    df_cos["example_num"] = df_cos["example_num"].astype(int)
    df_cos["cos_sim"] = df_cos["cos_sim"].astype(float)
    df_cos["cos_sim_log"] = np.log2(df_cos["cos_sim"] + 1)

    return df_cos


def run_leave_one_out(df_in, depvar, predvar):

    # fit
    d = {}
    tasks = list(df_in["task"].unique())

    for t in tasks:
        d[t] = {}
        train = df_in[df_in["task"] != t].reset_index(drop=True)
        test_heldout = df_in[df_in["task"] == t].reset_index(
            drop=True
        )  # one heldout task

        # fit the simple regression model
        lm = ols(f"""{depvar} ~ {predvar}""", train).fit()
        test_heldout["pred"] = (
            test_heldout[predvar] * lm.params[predvar].item()
            + lm.params["Intercept"].item()
        )

        d[t]["slope"] = lm.params[predvar].item()
        d[t]["intercept"] = lm.params["Intercept"].item()
        d[t]["pred"] = test_heldout["pred"].item()
        d[t]["depvar"] = test_heldout[depvar].item()

        # d[t]['pred_data'] = 1 / d[t]['pred']
        # d[t]['actual_data'] = 1 / d[t]['depvar']
        # d[t]['data_diff'] = d[t]['actual_data'] - d[t]['pred_data']

        d[t]["abs_diff"] = abs(test_heldout[depvar] - test_heldout["pred"]).item()
        d[t]["diff"] = (test_heldout[depvar] - test_heldout["pred"]).item()
        d[t]["sqrd_err"] = ((test_heldout[depvar] - test_heldout["pred"]) ** 2).item()

    res = pd.DataFrame(d).T.reset_index().rename(columns={"index": "task"})

    return res


def get_model_prefix(model_name):

    model_prefix = None
    if "mistral" in model_name.lower():
        model_prefix = "mistral"
    elif "llama" in model_name.lower():
        model_prefix = "llama"
    elif "qwen" in model_name.lower():
        model_prefix = "qwen"
    elif "smollm" in model_name.lower():
        model_prefix = "smollm"

    return model_prefix


def check_nan_in_pytree(pytree: dict | list | tuple | torch.Tensor, location: str = ""):
    print(f"Checking nan in {location}")
    found = _check_nan_in_pytree(pytree, location)
    if not found:
        print(f"No nan found in {location}")
    return found


def _check_nan_in_pytree(
    pytree: dict | list | tuple | torch.Tensor, location: str = ""
) -> bool:
    found = False
    if isinstance(pytree, dict):
        for key, value in pytree.items():
            found_in_value = _check_nan_in_pytree(value, f"{location}.{key}")
            found = found or found_in_value
    elif isinstance(pytree, list):
        for i, value in enumerate(pytree):
            found_in_value = _check_nan_in_pytree(value, f"{location}.{i}")
            found = found or found_in_value
    elif isinstance(pytree, tuple):
        for i, value in enumerate(pytree):
            found_in_value = _check_nan_in_pytree(value, f"{location}.{i}")
            found = found or found_in_value
    elif isinstance(pytree, torch.Tensor):
        if torch.isnan(pytree).any() or torch.isinf(pytree).any():
            print(f"Nan found in {location}")
            found = True
    elif isinstance(pytree, str | int):
        pass
    elif isinstance(pytree, float):
        if torch.isnan(pytree) or torch.isinf(pytree):
            print(f"Nan found in {location}")
            found = True
    else:
        print(f"Unk type {type(pytree)} in {location}")

    return found


def custom_collator_completion_only(tokenizer, assistant_start_token, examples):

    # Tokenize each example
    tokenized_examples = []
    prompt_lengths = []

    # Build exclude token IDs from the tokenizer (eos + newline after header)
    exclude_tok_ids = []
    if tokenizer.eos_token_id is not None:
        exclude_tok_ids.append(tokenizer.eos_token_id)
    # Add the newline token that follows the assistant header
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline_ids) == 1:
        exclude_tok_ids.append(newline_ids[0])

    for ex in examples:
        if isinstance(ex, str):
            message = ex
        else:
            message = ex['messages']

        # Tokenize the full message
        tokenized = tokenizer(message, add_special_tokens=False)

        # Tokenize just the prompt part (before assistant)
        prompt = message.split(assistant_start_token)[0] + assistant_start_token
        prompt_tokenized = tokenizer(prompt, add_special_tokens=False)
        prompt_lengths.append(len(prompt_tokenized['input_ids']))

        # Add to list
        tokenized_examples.append({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids'].copy(),  # Copy for labels
        })

    # Collate and pad
    batch = default_data_collator(tokenized_examples)

    # Mask labels before assistant response
    for i, prompt_len in enumerate(prompt_lengths):
        batch["labels"][i, :prompt_len] = -100
        exclude_tok_idxs = sum([batch['labels'] == i for i in exclude_tok_ids]).bool()
        batch['labels'][exclude_tok_idxs] = -100

    return batch
