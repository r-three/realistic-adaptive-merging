from dataclasses import dataclass, field

from peft import PeftConfig, PeftType


@dataclass
class ArrowConfig(PeftConfig):
    """Configuration class for Arrow routing over a library of LoRA experts.

    Arrow computes a prototype vector per expert via SVD of the expert's LoRA
    weights, then routes each input token to the most aligned experts using
    absolute dot-product similarity.
    """

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
    # Arrow-specific parameters
    top_k: int = field(default_factory=lambda: -1)
    # Number of experts per token (-1 = soft routing over all experts)
    router_temp: float = field(default_factory=lambda: 1.0)
    # Temperature for softmax routing (higher = softer distribution)

    def __post_init__(self):
        self.peft_type = PeftType.POLY
