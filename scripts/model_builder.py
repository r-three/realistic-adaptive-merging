"""
Model Builder Utilities

Contains utilities for building models with optional PEFT configurations.
"""

from scripts.llama_child import Llama
from scripts.qwen_child import Qwen
from weight_processing import get_peft_config


def get_model_prefix(model_name):

    model_prefix = None
    if "llama" in model_name.lower():
        model_prefix = "llama"
    elif "qwen" in model_name.lower():
        model_prefix = "qwen"

    return model_prefix


def get_peft_model_prefix(model_name):

    peft_config = get_peft_config(model_name)
    if peft_config is not None and peft_config.base_model_name_or_path is not None:
        base_model_name = peft_config.base_model_name_or_path
    else:
        base_model_name = model_name

    return get_model_prefix(base_model_name)


def build_model(args, peft_config=None):
    """Build model, tokenizer and subclass from args and optionally peft_config.
    If peft config is None, the model_name will be loaded. If that model_name point to a peft model, the function will still output the saved peft model.
    If peft config is specified, the model_name need to be a base model, and the peft model will be built from the base model and the peft_config.
    """

    model_prefix = get_model_prefix(args.model_name) or get_peft_model_prefix(
        args.model_name
    )
    if model_prefix == "llama":
        subclass = Llama
    elif model_prefix == "qwen":
        subclass = Qwen
    else:
        raise ValueError(
            f"Unsupported model prefix {model_prefix} from model {args.model_name}"
        )

    model_cls = subclass(
        model_name=args.model_name,
        task=args.task,
        use_quantized=args.use_quantized,
        seed=args.seed,
        peft_config=peft_config,
    )
    model, tokenizer = model_cls.get_model_and_tokenizer(
        tokenizer_path=getattr(args, "tokenizer_path", None),
        use_flash_attention=getattr(args, "use_flash_attention", False),
        use_safetensors=getattr(args, "use_safetensor", False),
    )
    return model_cls, model, tokenizer
