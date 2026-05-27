from dataclasses import dataclass, field
from typing import Literal
from peft import PeftConfig, PeftType


@dataclass
class LorahubConfig(PeftConfig):
    """Configuration class for gradient-based Lorahub"""

    expert_info_dir: str = field(default_factory=lambda: "")
    expert_list_path: str = field(default_factory=lambda: "")
    # list of loaded expert names
    expert_list: list[str] = field(default_factory=lambda: [])
    # list of loaded expert ranks
    rank: list[int] = field(default_factory=lambda: [])
    # list of loaded expert scaling factor (alpha / rank)
    scaling: list[float] = field(default_factory=lambda: [])
    # total number of loaded experts
    num_experts: int = field(default_factory=lambda: -1)
    # union of target modules of the loaded experts
    target_modules: list[str] = field(default_factory=lambda: [])
    router_activation: str = field(default_factory=lambda: "linear")
    router_weight_init_method: str = field(default_factory=lambda: "uniform")
    include_activation_coeff: bool = field(default_factory=lambda: True)
    pi_tuning: bool = field(default_factory=lambda: False)
    weight_granularity: str = field(default_factory=lambda: "per_module")
    force_routing: None | float = field(default_factory=lambda: None)
    reinit_case: str = field(default_factory=lambda: "")

    def __post_init__(self):
        self.peft_type = PeftType.POLY
        if self.pi_tuning:
            assert self.weight_granularity == "per_module"
            assert self.router_activation == "softmax"
            assert self.router_weight_init_method == "equal_weight"
