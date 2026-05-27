import argparse
import os
from peft_extension.lorahub.config import LorahubConfig
from peft_extension.arrow.config import ArrowConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import gc
import torch
import logging
from transformers import set_seed, EarlyStoppingCallback, TrainerCallback
from utils.logging_utils import setup_logging
from scripts.evaluation import evaluate_model
from collections import OrderedDict
import json
import time
from collections import Counter
import shutil
import glob
from scripts.model_builder import build_model
import torch.nn.functional as F


def get_optimizer(model, args) -> torch.optim.Optimizer:
    all_params = [
        (name, param) for name, param in model.named_parameters() if param.requires_grad
    ]
    named_param_groups = {
        "routing": {
            "params": [],
            "lr": args.lr_multiplier_routing * args.lr,
        },
        "sigma": {"params": [], "lr": args.lr_multiplier_sigma * args.lr},
        "matrix_shared": {
            "params": [],
            "lr": args.lr_multiplier_matrix_shared * args.lr,
        },
        "matrix_new": {
            "params": [],
            "lr": args.lr_multiplier_matrix_new * args.lr,
        },
        "gate": {"params": [], "lr": args.lr_multiplier_gate * args.lr},
        "else": {"params": [], "lr": args.lr_multiplier_else * args.lr},
    }
    for name, param in all_params:
        if "poly_shared_" in name:
            named_param_groups["routing"]["params"].append(param)
        elif "fauxpoly_prototypes" in name:
            named_param_groups["routing"]["params"].append(param)
        elif "routing" in name:
            named_param_groups["routing"]["params"].append(param)
        elif "sigma" in name:
            named_param_groups["sigma"]["params"].append(param)
        elif "shared_A" in name or "shared_B" in name:
            named_param_groups["matrix_shared"]["params"].append(param)
        elif "lora_A" in name or "lora_B" in name or "new_A" in name or "new_B" in name:
            named_param_groups["matrix_new"]["params"].append(param)
        elif "gate" in name:
            named_param_groups["gate"]["params"].append(param)
        else:
            named_param_groups["else"]["params"].append(param)

    for group_name in list(named_param_groups.keys()):
        if len(named_param_groups[group_name]["params"]) == 0:
            del named_param_groups[group_name]
        elif named_param_groups[group_name]["lr"] == 0:
            del named_param_groups[group_name]

    for group_name in list(named_param_groups.keys()):
        print(
            f"{group_name}: {len(named_param_groups[group_name]['params'])} trainable param tensors with lr {named_param_groups[group_name]['lr']}"
        )
        print(
            f"    Shapes: {Counter([p.shape for p in named_param_groups[group_name]['params']])}"
        )

    optimizer = torch.optim.AdamW(
        named_param_groups.values(),
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    return optimizer


class CustomSFTTrainer(SFTTrainer):
    """Custom trainer that logs per-token losses for first val batch and first train step after val."""

    def __init__(self, regularization_coeff, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_step_count = 0
        self.train_step_after_eval = False
        self.regularization_coeff = regularization_coeff

    def compute_loss(self, model, *args, **kwargs):
        """Override to ensure labels are passed to the model for loss computation."""
        compute_loss_output = super().compute_loss(model, *args, **kwargs)

        if self.regularization_coeff is not None and self.regularization_coeff > 0:
            trainable_1d_params = []
            for param in model.parameters():
                if param.requires_grad and max(param.shape) == param.numel():
                    trainable_1d_params.append(param)

            if len(trainable_1d_params) > 0:
                l1_reg = torch.sum(
                    torch.stack([torch.sum(torch.abs(p)) for p in trainable_1d_params])
                ) / len(trainable_1d_params)
                l1_loss = self.regularization_coeff * l1_reg

                if isinstance(compute_loss_output, tuple):
                    compute_loss_output = (
                        compute_loss_output[0] + l1_loss,
                    ) + compute_loss_output[1:]
                else:
                    compute_loss_output += l1_loss
                if self.state.global_step % self.args.logging_steps == 0:
                    print(f"L1 regularization loss: {l1_loss.item():.6f}")

        return compute_loss_output

    def prediction_step(self, model, inputs, *args, **kwargs):
        """Override to log per-token loss for first eval batch."""
        if self.eval_step_count == 0:
            self._log_per_token_loss(model, inputs, "VALIDATION")
            self.train_step_after_eval = True

        self.eval_step_count += 1
        output = super().prediction_step(model, inputs, *args, **kwargs)
        return output

    def training_step(self, model, inputs, *args, **kwargs):
        """Override to log per-token loss for first train step after eval."""
        if self.train_step_after_eval:
            self._log_per_token_loss(model, inputs, "TRAINING")
            self.train_step_after_eval = False
            self._log_first_param_value(model)

        return super().training_step(model, inputs, *args, **kwargs)

    def _log_first_param_value(self, model):
        """Log the value of the first parameter tensor."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(
                    f"First trainable parameter: {name}, values: {param.data.view(-1)[:10]}"
                )
                break

    def evaluate(self, *args, **kwargs):
        """Reset eval step count before each evaluation."""
        self.eval_step_count = 0
        return super().evaluate(*args, **kwargs)

    def _log_per_token_loss(self, model, inputs, phase):
        """Compute and log per-token losses with detokenized tokens."""
        was_training = model.training
        model.eval()

        selected_idx = 0
        with torch.no_grad():
            input_ids = inputs["input_ids"][selected_idx : selected_idx + 1]
            labels = inputs["labels"][selected_idx : selected_idx + 1]

            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            tokens = input_ids[0].tolist()
            token_strs = [self.processing_class.decode([t]) for t in tokens]

            print(f"\n{'='*80}")
            print(
                f"PER-TOKEN LOSS - STEP {self.state.global_step} {phase} BATCH (First Example)"
            )
            print(f"{'='*80}")
            print(
                f"{'Position':<10} {'Input Token':<30} {'Next Token':<30} {'Loss':<12}"
            )
            print(f"{'-'*80}")

            for i, (input_str, target_str, loss_val) in enumerate(
                zip(token_strs[:-1], token_strs[1:], per_token_loss)
            ):
                if shift_labels[0, i] != -100:
                    print(
                        f"{i:<10} {repr(input_str):<30} {repr(target_str):<30} {loss_val.item():<12.6f}"
                    )

            mean_loss = per_token_loss[shift_labels[0] != -100].mean().item()
            print(f"{'-'*80}")
            print(f"Mean Loss: {mean_loss:.6f}")
            print(f"{'='*80}\n")

        if was_training:
            model.train()


def main(args: argparse.Namespace, logger: logging.Logger):
    logger.info("Starting main function")
    logger.info("Args: \n" + str(args))
    job_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(job_dir, exist_ok=True)
    time0 = time.time()

    gc.collect()
    torch.cuda.empty_cache()

    resume_from_checkpoint = args.resume_from_checkpoint

    training_arguments = SFTConfig(
        # save dir
        output_dir=job_dir,
        do_eval=True,
        eval_on_start=True,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=args.log_and_save_step,
        logging_first_step=True,
        # save_only_model=True, # do not save optimizer state etc, takes up space
        logging_steps=args.log_and_save_step,  # same as eval_steps
        save_total_limit=1,  # only save best one + last checkpoint
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=args.run_name,
        num_train_epochs=args.epoch,
        max_steps=args.step,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.effective_batch_size // args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        metric_for_best_model="loss",
        max_length=args.max_seq_len,
        disable_tqdm=True,
        save_only_model=True,
        assistant_only_loss=True,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    logger.info("Training_arguments: \n" + str(training_arguments))
    # setting seed and adding CUDA related dependencies
    set_seed(args.seed)

    # init model class that has initialization methods
    if args.peft_type == "lorahub":
        peft_config = LorahubConfig(
            expert_info_dir=args.expert_info_dir,
            expert_list_path=args.expert_list_path,
            task_type="CAUSAL_LM",
            router_activation=args.router_activation,
            router_weight_init_method=args.router_weight_init_method,
            include_activation_coeff=args.include_activation_coeff,
            weight_granularity=args.weight_granularity,
            pi_tuning=args.pi_tuning,
            force_routing=args.force_routing,
            reinit_case=args.reinit_case,
        )
    elif args.peft_type == "arrow":
        peft_config = ArrowConfig(
            expert_info_dir=args.expert_info_dir,
            expert_list_path=args.expert_list_path,
            task_type="CAUSAL_LM",
            top_k=args.top_k,
            router_temp=args.router_temp,
        )
    elif args.peft_type == "lora":
        # Translate shorthand to actual module names
        module_map = {
            "k": "k_proj",
            "q": "q_proj",
            "v": "v_proj",
            "o": "o_proj",
            "g": "gate_proj",
            "d": "down_proj",
            "u": "up_proj",
        }
        target_modules = [
            module_map[char] for char in args.lora_target_modules if char in module_map
        ]

        peft_config = LoraConfig(
            target_modules=target_modules,
            r=args.rank,
            lora_alpha=16,
            lora_dropout=args.lora_dropout,
            task_type="CAUSAL_LM",
        )
    elif args.peft_type == "none":
        peft_config = None
    else:
        raise ValueError(f"Unknown peft_type: {args.peft_type}")

    model_cls, model, tokenizer = build_model(args, peft_config)

    total_trainable = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  trainable: {name}, shape={tuple(param.shape)}, numel={param.numel()}")
            total_trainable += param.numel()
    print(f"Total trainable parameters: {total_trainable:,}")

    data = model_cls.get_data(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        filter_long_seq=False,  ## True, # jje; set this flag to True, because otherwise the run fails and results in NaN loss -> but unset to False again
        data_size=args.data_size,
        subset_sample_path=args.subset_sample_path,
        swap_train_val=args.swap_train_val,
        combine_train_valid=args.combine_train_valid,
    )

    train_split = "train" if not args.combine_train_valid else "train+valid"
    valid_split = "valid" if not args.combine_train_valid else "train+valid"

    logger.info(
        f"train count: {data[train_split].num_rows}, valid count: {data[valid_split].num_rows}, test count: {data['test'].num_rows}"
    )

    trainer = CustomSFTTrainer(
        model=model,
        train_dataset=data[train_split],
        eval_dataset=data[valid_split],
        peft_config=None,
        args=training_arguments,
        processing_class=tokenizer,
        # data_collator=model_cls.get_datacollator(
        #     tokenizer=tokenizer, completion_only=True
        # ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
        optimizers=(
            get_optimizer(model, args),
            None,
        ),  # Pass None for scheduler, the trainer will create scheduler accounting for multiple param groups
        regularization_coeff=args.regularization_coeff,
    )
    trainer._signature_columns = [
        "attention_mask",
        "cache_position",
        "input_ids",
        "inputs_embeds",
        "kwargs",
        "label",
        "label_ids",
        "labels",
        "logits_to_keep",
        "output_attentions",
        "output_hidden_states",
        "past_key_values",
        "position_ids",
        "use_cache",
    ]  # Don't ask me why this is needed. Merely seeing it already hurts my sanity.

    # handle PEFT + FSDP
    if peft_config is not None:
        if getattr(trainer.accelerator.state, "fsdp_plugin", None):
            from peft.utils.other import fsdp_auto_wrap_policy

            logger.info("fsdp auto wrap policy to handle PEFT model")
            fsdp_plugin = trainer.accelerator.state.fsdp_plugin
            fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    if args.step > 0:
        logger.info("Training...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if trainer.is_fsdp_enabled:
        logger.info("is_fsdp_enabled is true, will save the model")
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    if args.save_model:
        trainer.save_model(
            output_dir=os.path.join(job_dir, "final_model"),
        )

    logger.info("Evaluating...")
    output_file = os.path.join(job_dir, "predictions.csv")

    generation_results, metrics = evaluate_model(
        args,
        model_cls,
        model,
        tokenizer,
        data,
        split=args.eval_split.split(","),
        output_file=output_file,
    )
    logger.info(f"Evaluation table first five rows: \n{generation_results.head()}")
    logger.info(f"Metrics: \n{metrics}")

    # Save LoraHub-specific results if using lorahub
    # if args.peft_type == "lorahub" and peft_config is not None:
    #     lorahub_result = {"expert_list": peft_config.expert_list}
    #     if hasattr(model, "poly_shared_expert_weights"):
    #         lorahub_result["weights"] = model.poly_shared_expert_weights.data[
    #             0
    #         ].tolist()
    #     lorahub_result.update(metrics)

    #     lorahub_result_path = os.path.join(job_dir, "lorahub_result.json")
    #     with open(lorahub_result_path, "w") as f:
    #         json.dump(lorahub_result, f, indent=4)
    #     logger.info(f"Saved LoraHub results to {lorahub_result_path}")

    job_summary = OrderedDict([(k, v) for k, v in args.__dict__.items()])
    job_summary.update(metrics)
    job_summary["final_validation_loss"] = trainer.state.best_metric

    # Add best validation loss from trainer state
    if args.step > 0 and hasattr(trainer.state, "best_metric"):
        job_summary["best_validation_loss"] = trainer.state.best_metric
        job_summary["best_global_step"] = trainer.state.best_global_step

    time1 = time.time()
    logger.info(f"Time taken: {time1 - time0:.2f} seconds")
    job_summary["time_taken"] = (time1 - time0) / 3600  # convert to hours
    with open(os.path.join(job_dir, "job_summary.json"), "w") as f:
        json.dump(job_summary, f)
    print("Job summary: \n" + json.dumps(job_summary, indent=4))

    checkpoint_dirs = glob.glob(os.path.join(job_dir, "checkpoint*"))
    for checkpoint_dir in checkpoint_dirs:
        logger.info(f"Removing checkpoint directory: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)

    # jje: temporary logic to remove expert_info.pt
    if args.expert_info_dir != None:
        logger.info("Removing expert_info dir")
        try:
            shutil.rmtree(args.expert_info_dir)
        except:
            print(f"Failed to remove {args.expert_info_dir}")

def build_parser():
    parser = argparse.ArgumentParser(description="Tuning Job")

    # logging and path
    parser.add_argument("--run_name", type=str, default="tuning_debug")
    parser.add_argument("--output_dir", type=str, default="outputs/")

    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="model checkpoint from huggingface",
    )
    parser.add_argument(
        "--peft_type",
        type=str,
        default="lorahub",
        choices=["lora", "lorahub", "arrow", "none"],
        help="PEFT type",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=-1,
        help="rank of lora model",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="kqvogdu",
        help="target modules for lora. Use shorthand: k=k_proj, q=q_proj, v=v_proj, o=o_proj, g=gate_proj, d=down_proj, u=up_proj",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="dropout rate for lora layers",
    )
    parser.add_argument("--router_activation", type=str, default="linear")
    parser.add_argument("--router_base_as_an_expert", action="store_true")
    parser.add_argument("--expert_sigma_normalization", action="store_true")
    parser.add_argument(
        "--num_experts",
        type=int,
        default=-1,
        help="number of experts",
    )
    parser.add_argument(
        "--force_routing",
        type=float,
        default=None,
        help="force the model to route to the task expert, set to None to disable",
    )
    parser.add_argument(
        "--single_sigma",
        action="store_true",
        help="use a single sigma for all experts",
    )
    parser.add_argument(
        "--sigma_gate",
        type=str,
        default="none",
        choices=["none", "sigmoid", "tanh"],
        help="type of gate to apply to sigma",
    )
    parser.add_argument(
        "--apply_ties",
        type=str,
        default="none",
        choices=["none", "majority", "target"],
        help="whether to apply ties to sigmas, and how to select sign",
    )

    # lorahub-specific arguments
    parser.add_argument(
        "--expert_info_dir",
        type=str,
        default=None,
        help="directory containing expert info for lorahub",
    )
    parser.add_argument(
        "--expert_list_path",
        type=str,
        default=None,
        help="path to expert list file for lorahub (alternative to expert_info_dir)",
    )
    parser.add_argument(
        "--weight_granularity",
        type=str,
        choices=[
            "per_expert",
            "per_layer",
            "per_sublayer",
            "per_module",
            "per_dimension",
        ],
        default="per_module",
        help="granularity of weight training for lorahub",
    )
    parser.add_argument(
        "--router_weight_init_method",
        type=str,
        default="equal_weight",
        help="router weight initialization method for lorahub",
    )
    parser.add_argument(
        "--include_activation_coeff",
        action="store_true",
        help="whether to include a, b term for activation in lorahub",
    )
    parser.add_argument(
        "--pi_tuning",
        action="store_true",
        help="whether to use pi tuning in lorahub",
    )
    parser.add_argument(
        "--reinit_case",
        type=str,
        default="",
        help="reinit case for lorahub",
    )

    # arrow-specific arguments
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="number of experts per token for arrow routing (-1 = soft routing over all)",
    )
    parser.add_argument(
        "--router_temp",
        type=float,
        default=1.0,
        help="temperature for arrow softmax routing (higher = softer distribution)",
    )

    # data
    parser.add_argument("--task", default=None, type=str, help="finetuning task")
    parser.add_argument(
        "--subset_sample_path",
        type=str,
        default="auto",
        help="only train the model on specifically designated subset indexable by the unique task id(default now is incorrect examples)",
    )
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
        default=None,
        help="number of available examples for train and valid",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="valid,test",
        help="which split to use for evaluation, train, valid or test",
    )
    parser.add_argument(
        "--swap_train_val",
        action="store_true",
        help="whether to swap train and validation sets",
    )
    parser.add_argument(
        "--combine_train_valid",
        action="store_true",
        help="combine train and valid splits into a single train+valid split",
    )
    # training
    parser.add_argument("--lr", default=5e-5, type=float, help="learning rate")
    parser.add_argument(
        "--warmup_ratio",
        default=0.1,
        type=float,
        help="warmup ratio as a percentage of the training steps",
    )
    parser.add_argument("--epoch", default=40, type=int, help="total training epoch")
    parser.add_argument(
        "--step", default=500, type=int, help="total number of steps, override epoch"
    )
    parser.add_argument(
        "--batch_size", default=8, type=int, help="batch size for training and eval"
    )
    parser.add_argument(
        "--eval_batch_size",
        default=4,
        type=int,
        help="evaluation batch size",
    )
    parser.add_argument(
        "--effective_batch_size",
        default=8,
        type=int,
        help="effective batch size for training",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="weight decay for AdamW optimizer",
    )

    parser.add_argument(
        "--use_quantized",
        action="store_true",
        help="whether to load quantized base model",
    )
    parser.add_argument(
        "--ds_config",
        type=str,
        default="ds_configs/zero2_config.json",
        help="deepspeed config path relative to this script",
    )
    parser.add_argument(
        "--bf16", action="store_true", help="argument passed to the trainer if bf16"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="if the trainer has halted, check for the checkpoint folder and continue from there",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="whether to use flash attention or not",
    )
    parser.add_argument(
        "--regularization_coeff",
        type=float,
        default=None,
        help="coefficient for L1 regularization on 1D parameters (routing weights, sigmas)",
    )
    parser.add_argument(
        "--log_and_save_step",
        type=int,
        default=10,
        help="evaluate run results and save every n steps",
    )
    parser.add_argument(
        "--lr_multiplier_matrix_shared",
        type=float,
        default=1.0,
        help="lr multiplier for lora matrix (shared_A, shared_B)",
    )
    parser.add_argument(
        "--lr_multiplier_matrix_new",
        type=float,
        default=1.0,
        help="lr multiplier for lora matrix (new_A, new_B)",
    )
    parser.add_argument(
        "--lr_multiplier_sigma",
        type=float,
        default=1.0,
        help="lr multiplier for lora matrix (sigma)",
    )
    parser.add_argument(
        "--lr_multiplier_routing",
        type=float,
        default=1.0,
        help="lr multiplier for lora matrix (routing weights)",
    )
    parser.add_argument(
        "--lr_multiplier_gate",
        type=float,
        default=1.0,
        help="lr multiplier for lora matrix (gate weights)",
    )
    parser.add_argument(
        "--lr_multiplier_else",
        type=float,
        default=0.0,
        help="lr multiplier for else (non-lora parameters)",
    )
    parser.add_argument(
        "--calculate_all_metrics",
        action="store_true",
        help="calculate all metrics",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="save the model to the output_dir",
    )

    parser.add_argument(
        "--valid_split",
        type=str,
        default="valid"
    )

    return parser


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    parser = build_parser()
    args = parser.parse_args()
    logger = setup_logging()
    main(args, logger)
