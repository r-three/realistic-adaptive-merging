import os
import logging
import copy
import torch
import pandas as pd
from tqdm import tqdm
from transformers import GenerationConfig
from torch.utils.data import DataLoader
from utils.calculate_eval_metrics import (
    calculate_bleu,
    calculate_rouge,
    calculate_exact_string_match,
    calculate_sequence_accuracy,
    calculate_squad,
)
from transformers import DataCollatorWithPadding


def get_dataloader(data, split, tokenizer, batch_size, max_seq_len):
    """Get dataloader for batched evaluation"""

    def tokenize_fn(example, idx):
        messages = example["messages"]
        assert messages[-1]["role"] == "assistant"
        result = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=True
        )
        result["data_idx"] = idx
        return result

    # Tokenize dataset with index tracking
    tokenized_data = data[split].map(tokenize_fn, with_indices=True)

    # Filter out examples that are too long
    tokenized_data = tokenized_data.filter(lambda x: len(x["input_ids"]) < max_seq_len)
    tokenized_data = tokenized_data.select_columns(
        ["input_ids", "attention_mask", "data_idx"]
    )

    mycollator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(
        tokenized_data, collate_fn=mycollator, batch_size=batch_size, shuffle=False
    )
    return loader


def evaluate_model(
    args,
    model_cls,
    model,
    tokenizer,
    data,
    split: str | list[str] = "test",
    output_file: str = None,
) -> tuple[pd.DataFrame | dict[str, pd.DataFrame], dict[str, float]]:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    print("device for evaluation: ", set([p.device for p in model.parameters()]))

    if isinstance(split, str):
        split = [split]

    all_metrics = {}
    all_results = {}

    # Evaluate on each split
    for split_name in split:
        # create a dataloader based on the loaded data
        myloader = get_dataloader(
            data=data,
            split=split_name,
            tokenizer=tokenizer,
            batch_size=args.eval_batch_size,
            max_seq_len=args.max_seq_len,
        )

        generation_results = eval_on_heldout(
            model=model,
            data=data,
            split=split_name,
            tokenizer=tokenizer,
            prompt=model_cls.task_args,
            dataloader=myloader,
            assistant_start_token=model_cls.assistant_start_token,
            logger=logging.getLogger(__name__),
            device=device,
        )

        metric_type = model_cls.task_args.get("metric", "exact_string_match")
        metrics = {}
        if metric_type == "exact_string_match":
            accuracy = calculate_exact_string_match(
                preds=generation_results["pred_clean"],
                refs=generation_results["target"],
            )
            metrics.update(accuracy)
        elif metric_type == "bleu":
            metrics.update(
                calculate_bleu(
                    preds=generation_results["pred_clean"],
                    refs=generation_results["target"],
                )
            )
        elif metric_type == "rouge":
            metrics.update(
                calculate_rouge(
                    preds=generation_results["pred_clean"],
                    refs=generation_results["target"],
                )
            )
        elif metric_type == "sequence_accuracy":
            metrics.update(
                calculate_sequence_accuracy(
                    preds=generation_results["pred_clean"],
                    refs=generation_results["target"],
                )
            )
        elif metric_type == "squad":
            metrics.update(
                calculate_squad(
                    preds=generation_results["pred_clean"].str.strip(),
                    refs=generation_results["target"].str.strip(),
                )
            )
        else:
            raise NotImplementedError(
                f"Metric {metric_type} not implemented. Please choose from 'exact_string_match', 'bleu', 'rouge', 'sequence_accuracy', or 'squad'."
            )

        # Add split prefix to metric keys
        prefixed_metrics = {f"{split_name}/{k}": v for k, v in metrics.items()}
        all_metrics.update(prefixed_metrics)

        # Add split column to identify which split each row came from
        generation_results["split"] = split_name
        all_results[split_name] = generation_results

    # Concatenate all results into a single DataFrame
    combined_results = pd.concat(all_results.values(), ignore_index=True)

    # Save predictions and references to file if output_file is specified
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        combined_results.to_csv(output_file, index=False)
        print(f"Saved evaluation results to {output_file}")

    return combined_results, all_metrics


def clean_model_output(out, assistant_start_token, eos_token):
    """Extract final model answer from model generation"""

    out = out.split(assistant_start_token)[1].split(eos_token)[
        0
    ]  # take the part after the first assistant prompt and before eos token
    if "####" in out:
        out = out.split("####")[
            -1
        ]  # a few datasets (gsm8k) with #### as the final answer design
    out = out.split("####")[
        -1
    ]  # a few datasets (gsm8k) with #### as the final answer designator; TODO: shouldn't affect other tasks; remove?

    return out


def clean_target_output(tar):
    """Extract final target answer from task dataset"""

    tar = str(tar)
    if "####" in tar:
        tar = tar.split("####")[
            -1
        ]  # a few datasets (gsm8k) with #### as the final answer design

    return tar


def eval_on_heldout(
    model,
    data,
    split,
    tokenizer,
    dataloader,
    prompt,
    assistant_start_token,
    logger,
    device,
):
    """Evaluate the loaded model on the task dataset."""

    d_res = {"target": [], "pred_clean": [], "pred_org": [], "example_id": []}

    target_key = prompt["assistant_output"]
    answer_len = prompt["answer_len"]

    # Get answer length from the data
    if answer_len == "short":
        max_new_toks = 30
    elif answer_len == "long":
        max_new_toks = 500

    # TODO: this generation config may have to change for non-clsasification task
    generation_config = GenerationConfig(
        # top_k=2,
        # temperature=0,
        max_new_tokens=max_new_toks,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating on heldout"):
            batch_in = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask"]
            }

            outputs = model.generate(
                **batch_in, generation_config=generation_config, tokenizer=tokenizer
            )
            response = tokenizer.batch_decode(outputs)
            response_clean = [
                clean_model_output(
                    out=r,
                    assistant_start_token=assistant_start_token,
                    eos_token=tokenizer.eos_token,
                )
                for r in response
            ]
            target_clean = [
                clean_target_output(data[split][int(idx)][target_key])
                for idx in batch["data_idx"]
            ]
            example_ids = [
                data[split][int(idx)]["example_id"] for idx in batch["data_idx"]
            ]
            d_res["target"] += target_clean
            d_res["pred_clean"] += response_clean
            d_res["pred_org"] += response
            d_res["example_id"] += example_ids

    print(f"Sample prediction:")
    for i in range(3):
        print(f"Example ID: {d_res['example_id'][i]}")
        print(f"Target: {d_res['target'][i]}")
        print(f"Predicted (clean): {d_res['pred_clean'][i]}")
        print(f"Predicted (original): {d_res['pred_org'][i]}")
        print("-" * 20)
    generation_results = pd.DataFrame(d_res, dtype=str)

    return generation_results
