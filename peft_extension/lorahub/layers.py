import torch
import os
import numpy as np
from torch import nn
from .config import LorahubConfig
from peft.tuners.tuners_utils import BaseTunerLayer


class LorahubModule(BaseTunerLayer):

    def __init__(self, peft_config: LorahubConfig, adapter_info: dict):
        # TODO 1: dtype and device info can be handled better instead of being hard-coded here -- have it follow model type?

        super().__init__()
        self.peft_config = peft_config

        # additional model parameter
        self.lora_A = nn.Parameter(adapter_info["lora_A"])
        self.lora_B = nn.Parameter(adapter_info["lora_B"])
        # metadata required for forward computation
        self.expert_indices = adapter_info["index"]
        self.r = peft_config.rank
        self.scaling = peft_config.scaling
        self.router_activation = peft_config.router_activation
        self.num_experts = peft_config.num_experts
        self.router_weight_init_method = peft_config.router_weight_init_method
        self.include_activation_coeff = peft_config.include_activation_coeff
        self.per_dimension = peft_config.weight_granularity == "per_dimension"


class LorahubLinear(nn.Module, LorahubModule):

    def __init__(
        self,
        target_base_layer: nn.Linear,
        peft_config: LorahubConfig,
        adapter_info: dict,
    ):
        super().__init__()
        LorahubModule.__init__(self, peft_config, adapter_info)

        self.target_base_layer = target_base_layer
        base_layer_weight = self.target_base_layer.weight.data
        self.device = base_layer_weight.device
        self.dtype = base_layer_weight.dtype

        self.lora_A.data = self.lora_A.data.to(device=self.device, dtype=self.dtype)
        self.lora_B.data = self.lora_B.data.to(device=self.device, dtype=self.dtype)
        # if reinit_case == none for pi_tuning, do not reinitialize the target lora
        if peft_config.pi_tuning and peft_config.reinit_case != "none":
            first_lora_start, first_lora_end = self.expert_indices[0]
            self.lora_A.data[first_lora_start:first_lora_end, :].normal_(
                mean=0, std=0.01
            )
            self.lora_B.data[:, first_lora_start:first_lora_end].zero_()
        if peft_config.reinit_case == "non_target_experts":
            first_lora_start, first_lora_end = self.expert_indices[0]
            # Calculate std from the target expert (index 0)
            target_A = self.lora_A.data[first_lora_end:, :]
            target_B = self.lora_B.data[:, first_lora_end:]
            std_A = target_A.std().item()
            std_B = target_B.std().item()

            self.lora_A.data[first_lora_end:, :].normal_(mean=0, std=std_A)
            self.lora_B.data[:, first_lora_end:].normal_(mean=0, std=std_B)
            print(
                f"Initialized non target experts with random weights (std_A={std_A}, std_B={std_B})"
            )
        elif peft_config.reinit_case == "all_experts":
            # Calculate std from all experts
            std_A = self.lora_A.data.std().item()
            std_B = self.lora_B.data.std().item()

            self.lora_A.data.normal_(mean=0, std=std_A)
            self.lora_B.data.normal_(mean=0, std=std_B)
            print(
                f"Initialized all experts with random weights (std_A={std_A}, std_B={std_B})"
            )
        else:
            pass
        self.poly_activation = self.set_activation_function()

        num_routing_params = (
            self.lora_A.shape[0] if self.per_dimension else self.num_experts
        )
        # initialize the shared expert weights
        if self.router_weight_init_method == "normal":
            self.poly_shared_expert_weights = nn.Parameter(
                torch.empty(1, num_routing_params).normal_(mean=0, std=0.5)
            )
        elif self.router_weight_init_method == "uniform":
            self.poly_shared_expert_weights = nn.Parameter(
                torch.rand(1, num_routing_params)
            )
        elif self.router_weight_init_method == "equal_weight":
            self.poly_shared_expert_weights = nn.Parameter(
                torch.ones(1, num_routing_params) / (peft_config.num_experts**0.5)
            )
        elif self.router_weight_init_method == "target":
            weight = torch.zeros(1, num_routing_params)

            if peft_config.force_routing is not None:
                coeff = peft_config.force_routing
            else:
                coeff = 1.0
            if self.per_dimension:
                lora_start, lora_end = self.expert_indices[0]
                weight[0, lora_start:lora_end].fill_(coeff)
            else:
                weight[0, 0].fill_(coeff)
            self.poly_shared_expert_weights = nn.Parameter(weight)
        else:
            raise ValueError(
                f"Unsupported router weight initialization method: {self.router_weight_init_method}"
            )
        self.poly_shared_expert_weights.data = self.poly_shared_expert_weights.data.to(
            device=self.device, dtype=self.dtype
        )

    def set_activation_function(self):
        if self.router_activation == "softmax" or self.router_activation == "sigmoid":
            if self.include_activation_coeff:
                self.poly_a_term = nn.Parameter(torch.ones(1, self.num_experts))
                self.poly_b_term = nn.Parameter(torch.zeros(1, self.num_experts))
                self.poly_a_term.data = self.poly_a_term.data.to(
                    device=self.device, dtype=self.dtype
                )
                self.poly_b_term.data = self.poly_b_term.data.to(
                    device=self.device, dtype=self.dtype
                )

            if self.router_activation == "softmax":
                return nn.Softmax(dim=1)
            elif self.router_activation == "sigmoid":
                return nn.Sigmoid()

        elif self.router_activation == "squared":
            return lambda x: x**2
        elif self.router_activation == "real_linear":
            return lambda x: x
        elif self.router_activation == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(
                f"Unsupported router activation function: {self.router_activation}"
            )

    def forward(self, x):
        # x: (bs, seq_len, in_features)
        # lora_A: (sum(r_i), in_features)
        # lora_B: (out_features, sum(r_i))

        poly_shared_expert_weights = self.poly_activation(
            self.poly_shared_expert_weights
        )
        if (
            self.router_activation in ["sigmoid", "softmax"]
            and self.include_activation_coeff
        ):
            poly_shared_expert_weights = (
                self.poly_a_term * poly_shared_expert_weights + self.poly_b_term
            )

        # middle_scaling: (sum(r_i), )
        middle_scaling = torch.zeros_like(self.lora_A[:, 0])
        for exp_idx in np.arange(self.num_experts):
            lora_start, lora_end = self.expert_indices[exp_idx]
            if self.per_dimension:
                middle_scaling[lora_start:lora_end] = (
                    poly_shared_expert_weights[0, lora_start:lora_end]
                    * self.scaling[exp_idx]
                )
            else:
                middle_scaling[lora_start:lora_end] = (
                    poly_shared_expert_weights[0, exp_idx] * self.scaling[exp_idx]
                )

        expert_out = torch.einsum(
            "...i,ri,r,or->...o", x, self.lora_A, middle_scaling, self.lora_B
        )

        return self.target_base_layer(x) + expert_out
