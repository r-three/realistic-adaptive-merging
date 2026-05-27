import torch
import os
import torch.nn as nn
from peft import PolyModel
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from .config import ArrowConfig
from .layers import ArrowLinear
from ..lorahub.model import load_expert_weights, concat_experts


class ArrowModel(BaseTuner):
    prefix: str = "poly_"
    # Boilerplate methods
    get_peft_config_as_dict = PolyModel.get_peft_config_as_dict
    enable_adapter_layers = PolyModel.enable_adapter_layers
    disable_adapter_layers = PolyModel.disable_adapter_layers

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if not hasattr(self.model, "gradient_checkpointing_enable"):
            raise ValueError(
                f"The model {self.model.__class__.__name__} does not support gradient checkpointing"
            )
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def __init__(self, model, peft_config, adapter_name: str = "default"):
        self.all_expert_info = self._load_expert_weight_and_metadata(
            peft_config.expert_info_dir, peft_config.expert_list_path
        )
        peft_config.rank = self.all_expert_info["rank"]
        peft_config.scaling = self.all_expert_info["scaling"]
        peft_config.num_experts = len(self.all_expert_info["rank"])
        peft_config.target_modules = self.all_expert_info["target_modules"]
        peft_config.expert_list = self.all_expert_info["expert_list"]

        super().__init__(model, peft_config, adapter_name)

        # required by the parent classes
        base_config = model.config
        self.config = base_config
        self.generation_config = model.generation_config

    def _load_expert_weight_and_metadata(
        self, expert_info_dir: str, expert_list_path: str, verbose: bool = True
    ):
        # If expert_list_path is provided, load experts on the fly
        if expert_list_path:
            print(f"Loading experts on the fly from {expert_list_path}")
            with open(expert_list_path, "r") as f:
                expert_name = f.read()

            lora_module_list = expert_name.strip().split("\n")
            cache = load_expert_weights(lora_module_list)
            concat_expert, expert_list = concat_experts(cache)

            all_expert_info = {
                "layers": concat_expert,
                "expert_list": expert_list,
                "rank": [cache[exp]["peft_config"].r for exp in expert_list],
                "lora_alpha": [
                    cache[exp]["peft_config"].lora_alpha for exp in expert_list
                ],
                "scaling": [
                    cache[exp]["peft_config"].lora_alpha / cache[exp]["peft_config"].r
                    for exp in expert_list
                ],
                "target_modules": list(
                    set().union(
                        *[
                            cache[exp]["peft_config"].target_modules
                            for exp in expert_list
                        ]
                    )
                    - set(["lm_head"])
                ),
            }
        else:
            expert_weight_path = os.path.join(expert_info_dir, "expert_info.pt")
            all_expert_info = torch.load(expert_weight_path)

        if verbose:
            print(
                f"Loaded expert info with {len(all_expert_info['expert_list'])} experts."
            )
            print(f"Target modules: {all_expert_info['target_modules']}")
            print(f"Expert ranks: {all_expert_info['rank']}")

        return all_expert_info

    def _prepare_adapter_config(self, peft_config, model_config):
        if peft_config.target_modules is None:
            if (
                model_config["model_type"]
                not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
            ):
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                    model_config["model_type"]
                ]
            )
        return peft_config

    @staticmethod
    def _check_target_module_exists(peft_config, key):
        return check_target_module_exists(peft_config, key)

    def _create_and_replace(
        self,
        peft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        lora_A = None
        lora_B = None
        lora_A_layer = None
        lora_B_layer = None
        expert_index = None

        current_key = ".layers." + current_key.split(".layers.", 1)[-1]
        if len(self.all_expert_info["layers"]) > 0:
            for k, v in self.all_expert_info["layers"].items():
                if current_key in k:
                    if "lora_A" in k:
                        lora_A_layer = k
                        lora_A = v["state_dict"]
                        expert_index = v["index"]
                    elif "lora_B" in k:
                        lora_B_layer = k
                        lora_B = v["state_dict"]
                if (
                    lora_A is not None
                    and lora_B is not None
                    and expert_index is not None
                ):
                    break
            if lora_A is None or lora_B is None:
                print(f"{current_key} not found!")
                return

            del self.all_expert_info["layers"][lora_A_layer]
            del self.all_expert_info["layers"][lora_B_layer]

        adapter_info = {"lora_A": lora_A, "lora_B": lora_B, "index": expert_index}
        new_module = self._create_new_module(
            peft_config=peft_config, target=target, adapter_info=adapter_info
        )
        self._replace_module(parent, target_name, new_module)

    @staticmethod
    def _replace_module(parent_module, child_name, new_module):
        setattr(parent_module, child_name, new_module)

    @staticmethod
    def _create_new_module(
        peft_config: ArrowConfig, target: nn.Module, adapter_info: dict
    ):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, nn.Linear):
            return ArrowLinear(target_base_layer, peft_config, adapter_info)
        else:
            raise NotImplementedError(
                f"Unsupported module type: {type(target_base_layer)}"
            )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for _, p in self.named_parameters():
            p.requires_grad = False
        for module_name, module in self.named_modules():
            if isinstance(module, ArrowLinear):
                module.fauxpoly_prototypes.requires_grad = True

    @property
    def device(self):
        return self.model.device

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def set_adapter_layers(self, enabled=True):
        pass

    def set_adapter(self, adapter_name):
        pass
