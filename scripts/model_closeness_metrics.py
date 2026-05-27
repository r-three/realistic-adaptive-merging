from itertools import chain
from typing import Literal
import torch


EPSILON = 1e-6


def get_lora_lists(candidate_model_states, target_model_states):
    lora_A_names = set(
        [
            k
            for k in chain(candidate_model_states.keys(), target_model_states.keys())
            if ".lora_A" in k
        ]
    )
    return lora_A_names


def l2_closeness(candidate_model_states, target_model_states):
    lora_A_names = get_lora_lists(candidate_model_states, target_model_states)

    total_distance = 0
    for lora_A_key in lora_A_names:
        # lora_Bc: (out_dim, r1)
        # lora_Ac: (r1, in_dim)
        # lora_Bt: (out_dim, r2)
        # lora_At: (r2, in_dim)
        lora_B_key = lora_A_key.replace(".lora_A", ".lora_B", 1)
        if lora_A_key in candidate_model_states:
            candidate_matrix = (
                candidate_model_states[lora_B_key].float() @ candidate_model_states[lora_A_key].float()
            )
        else:
            candidate_matrix = None
        if lora_A_key in target_model_states:
            target_matrix = (
                target_model_states[lora_B_key].float() @ target_model_states[lora_A_key].float()
            )
        else:
            target_matrix = None
        if candidate_matrix is not None and target_matrix is not None:
            total_distance += torch.norm(candidate_matrix - target_matrix) ** 2
        elif candidate_matrix is not None:
            total_distance += torch.norm(candidate_matrix) ** 2
        elif target_matrix is not None:
            total_distance += torch.norm(target_matrix) ** 2

    # Convert distance to closeness with negative sign (higher is better)
    if isinstance(total_distance, torch.Tensor):
        total_distance = total_distance.item()

    nl_distance = -torch.log(torch.tensor(total_distance) + 1).item()
    return nl_distance


def cosine_closeness(
    candidate_model_states: dict[str, torch.Tensor],
    target_model_states: dict[str, torch.Tensor],
    variant: Literal["cosine", "clamp", "abs", "quasi_fim", "clamp_pm"] = "cosine",
    aggregate: Literal["micro", "macro"] = "micro",
) -> float:
    lora_A_names = get_lora_lists(candidate_model_states, target_model_states)

    if aggregate == "micro":
        list_similarity = []
    elif aggregate == "macro":
        list_prod = []
        list_norm_candidate = []
        list_norm_target = []
    else:
        raise ValueError("Invalid aggregate method. Choose 'micro' or 'macro'.")

    for lora_A_key in lora_A_names:
        # lora_Bc: (out_dim, r1)
        # lora_Ac: (r1, in_dim)
        # lora_Bt: (out_dim, r2)
        # lora_At: (r2, in_dim)
        lora_B_key = lora_A_key.replace(".lora_A", ".lora_B", 1)
        norm_key = lora_A_key.replace(".lora_A", ".lora_norm", 1)
        if lora_A_key in candidate_model_states:
            lora_Ac = candidate_model_states[lora_A_key].float()
            lora_Bc = candidate_model_states[lora_B_key].float()
            if variant == "quasi_fim":
                dW_c = (lora_Bc @ lora_Ac)**2 # (out_dim, in_dim)
            elif variant in ["clamp", "clamp_pm", "abs"]:
                dW_c = lora_Bc @ lora_Ac
            if norm_key not in candidate_model_states:
                if variant == "quasi_fim":
                    candidate_model_states[norm_key] = torch.norm(dW_c) ** 2
                else:
                    candidate_model_states[norm_key] = torch.trace(
                        (lora_Ac @ lora_Ac.T) @ (lora_Bc.T @ lora_Bc)
                    )
        if lora_A_key in target_model_states:
            lora_At = target_model_states[lora_A_key].float()
            lora_Bt = target_model_states[lora_B_key].float()
            if variant == "quasi_fim":
                dW_t = (lora_Bt @ lora_At)**2 # (out_dim, in_dim)
            elif variant in ["clamp", "clamp_pm", "abs"]:
                dW_t = lora_Bt @ lora_At
            if norm_key not in target_model_states:
                if variant == "quasi_fim":
                    target_model_states[norm_key] = torch.norm(dW_t) ** 2
                else:
                    target_model_states[norm_key] = torch.trace(
                        (lora_At @ lora_At.T) @ (lora_Bt.T @ lora_Bt)
                    )

        if norm_key in candidate_model_states and norm_key in target_model_states:
            if variant == "cosine":
                prod = torch.trace((lora_Ac @ lora_At.T) @ (lora_Bt.T @ lora_Bc))
            elif variant == "quasi_fim":
                prod = (dW_c * dW_t).sum()
            elif variant in ["clamp", "clamp_pm", "abs"]:
                dW_dot = dW_c * dW_t
                # dW_dot = torch.einsum("ik,kj,il,lj->ij", lora_Bc, lora_Ac, lora_Bt, lora_At)
                if variant == "clamp":
                    prod = torch.clamp(dW_dot, min=0).sum()
                elif variant == "clamp_pm":
                    prod = torch.clamp(dW_dot.sum(), min=0)
                elif variant == "abs":
                    prod = torch.abs(dW_dot).sum()
            else:
                raise ValueError(f"Invalid variant: {variant}")
        else:
            prod = 0.0

        if aggregate == "micro":
            list_similarity.append(
                prod
                / torch.sqrt(
                    candidate_model_states.get(norm_key, 0)
                    * target_model_states.get(norm_key, 0)
                    + EPSILON
                )
            )
        elif aggregate == "macro":
            list_prod.append(prod)
            list_norm_candidate.append(candidate_model_states.get(norm_key, 0))
            list_norm_target.append(target_model_states.get(norm_key, 0))

    if aggregate == "micro":
        total_similarity = sum(list_similarity) / len(list_similarity)
    elif aggregate == "macro":
        total_similarity = sum(list_prod) / torch.sqrt(
            sum(list_norm_candidate) * sum(list_norm_target) + EPSILON
        )

    if variant == "quasi_fim":
        # Sqrt to make score less concentrated
        total_similarity = total_similarity**0.5

    # Return similarity directly (higher is better)
    if isinstance(total_similarity, torch.Tensor):
        total_similarity = total_similarity.item()
    return total_similarity
