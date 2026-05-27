import peft
from peft import PeftType

from .lorahub import LorahubConfig, LorahubModel
from .arrow import ArrowConfig, ArrowModel


def custom_model_wrapper(
    model, peft_config, adapter_name: str = "default", low_cpu_mem_usage=False
):
    assert (
        adapter_name == "default"
    ), "We don't support mixed model, then there is no need for name."
    if isinstance(peft_config, dict):
        peft_config = peft_config[adapter_name]
    elif isinstance(peft_config, (LorahubConfig, ArrowConfig)):
        pass
    else:
        raise ValueError(
            f"peft_config should be a dict of configs or a single config, got {type(peft_config)}"
        )
    if isinstance(peft_config, LorahubConfig):
        return LorahubModel(model, peft_config, adapter_name)
    elif isinstance(peft_config, ArrowConfig):
        return ArrowModel(model, peft_config, adapter_name)
    else:
        raise ValueError(f"Custom model type {peft_config} not supported.")


def custom_config_wrapper(*args, **kwargs):
    if "expert_info_dir" in kwargs:
        return LorahubConfig(*args, **kwargs)
    else:
        raise ValueError("Cannot determine config type from args and kwargs.")


peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING[PeftType.POLY] = custom_model_wrapper
peft.mapping.PEFT_TYPE_TO_TUNER_MAPPING[PeftType.POLY] = custom_model_wrapper
peft.mapping.PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.POLY] = custom_config_wrapper
# This is ugly, but PEFT has a lot of hard-coded variables that can't be changed easily.
# There is no clean way to add a new model type.
# POLY and prompt leanring are the only two PEFT types that the forward function won't be ignored.
# Our model is more similar to POLY, so we choose to overwrite it.
# Sorry, Edoardo, Lucas, and Alessandro. PEFT made us do this.

# In addition, PEFT determines which parameter to save by keywords in the parameter name, hard-coded outside Tuner implementation, can you believe it?
# Check peft.utils.save_and_load.get_peft_model_state_dict for more details.
# In our case, since the keyword for POLY model is "poly_", we name all the trainable child modules of the TunerLayer with the prefix "fauxpoly_" to make sure they are properly saved
# Check LoraMoELinearLayer as an example.

# With these modifications, we can follow the standard PEFT API, model = get_peft_model(model, peft_config


# TODO: maybe try to register the model properly

# from peft.utils import register_peft_method
# PeftType._extend_enum("MOOSE", "MOOSE")
# PeftType._extend_enum("LORAHUB", "LORAHUB")

# register_peft_method(name="moose", config_cls=MooseConfig, model_cls=MooseModel)
# register_peft_method(name="lorahub", config_cls=LorahubConfig, model_cls=LorahubModel)
