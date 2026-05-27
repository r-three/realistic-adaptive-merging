import torch
from torch import nn
from torch.nn import functional as F

from .config import ArrowConfig


def compute_arrow_prototypes(lora_A, lora_B, expert_indices):
    """Compute Arrow prototypes via low-rank SVD of each expert's LoRA weights.

    For each expert with lora_A_i (r, in) and lora_B_i (out, r), the effective
    weight is W_i = lora_B_i @ lora_A_i, shape (out, in). The prototype is the
    top right singular vector of W_i — the input direction most affected by the
    expert. We compute this efficiently via SVD of a small r x r matrix.

    Args:
        lora_A: Concatenated LoRA A matrices, shape (sum_r, in_features).
        lora_B: Concatenated LoRA B matrices, shape (out_features, sum_r).
        expert_indices: List of (start, end) tuples delimiting each expert.

    Returns:
        Tensor of shape (num_experts, in_features) containing prototype vectors.
    """
    in_features = lora_A.shape[1]
    prototypes = []
    for start, end in expert_indices:
        if start == end:
            prototypes.append(torch.zeros(in_features, dtype=lora_A.dtype))
            continue

        A_i = lora_A[start:end, :].float()  # (r, in_features)
        B_i = lora_B[:, start:end].float()  # (out_features, r)

        # W_i = B_i @ A_i has shape (out, in), rank <= r
        # Its SVD: W_i = U Sigma V^T, we want V[:, 0] (top right singular vector)
        # Efficient: SVD of the r x r matrix A_i @ A_i^T @ B_i^T @ B_i
        # or equivalently, SVD of B_i @ A_i projected to rank-r
        # Use: QR of A_i^T then SVD of small matrix
        Q, R = torch.linalg.qr(A_i.T)  # Q: (in, r), R: (r, r)
        C = B_i @ R.T  # (out, r)
        _, _, Vh = torch.linalg.svd(C, full_matrices=False)  # Vh: (r, r)
        prototype = Q @ Vh[0]  # (in_features,)

        prototypes.append(prototype.to(lora_A.dtype))

    return torch.stack(prototypes)  # (num_experts, in_features)


class ArrowLinear(nn.Module):

    def __init__(
        self,
        target_base_layer: nn.Linear,
        peft_config: ArrowConfig,
        adapter_info: dict,
    ):
        super().__init__()
        self.target_base_layer = target_base_layer

        device = target_base_layer.weight.data.device
        dtype = target_base_layer.weight.data.dtype

        # Expert LoRA weights (frozen, scaling baked into lora_B)
        lora_A = adapter_info["lora_A"].to(device=device, dtype=dtype)
        lora_B = adapter_info["lora_B"].to(device=device, dtype=dtype)
        self.expert_indices = adapter_info["index"]
        self.num_experts = peft_config.num_experts

        for i in range(self.num_experts):
            start, end = self.expert_indices[i]
            if start != end:
                lora_B[:, start:end] *= peft_config.scaling[i]

        self.lora_A = nn.Parameter(lora_A, requires_grad=False)
        self.lora_B = nn.Parameter(lora_B, requires_grad=False)
        self.top_k = peft_config.top_k
        self.router_temp = peft_config.router_temp

        # Precompute rank-dim -> expert index mapping
        rank_to_expert = torch.zeros(lora_A.shape[0], dtype=torch.long)
        for i, (start, end) in enumerate(self.expert_indices):
            if start != end:
                rank_to_expert[start:end] = i
        self.register_buffer("rank_to_expert", rank_to_expert)

        # Compute Arrow prototypes from expert weights (trainable router)
        prototypes = compute_arrow_prototypes(
            self.lora_A.data, self.lora_B.data, self.expert_indices
        ).to(device=device, dtype=dtype)
        self.fauxpoly_prototypes = nn.Parameter(prototypes)

    def forward(self, x):
        # x: (batch, seq_len, in_features)
        base_out = self.target_base_layer(x)

        # Per-token routing via absolute dot-product similarity
        logits = F.linear(x, self.fauxpoly_prototypes).abs() / self.router_temp

        # Routing weights: top-k hard selection or soft over all experts
        if 0 < self.top_k < self.num_experts:
            topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
            weights = torch.zeros_like(logits)
            weights.scatter_(-1, topk_indices, F.softmax(topk_logits, dim=-1))
        else:
            weights = F.softmax(logits, dim=-1)

        # Map per-expert weights to per-rank-dimension via precomputed index
        middle_scaling = weights[..., self.rank_to_expert]  # (batch, seq, sum_r)

        # h: (batch, seq, sum_r)
        h = F.linear(x, self.lora_A) * middle_scaling
        expert_out = F.linear(h, self.lora_B)

        return base_out + expert_out
