# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Literal, List, Tuple, Dict, Union

import gc
import os
import json
import torch

from weight_processing.collect_lora_experts import get_normalized_weights
from peft import LoraConfig


def reshape_weight_task_tensors(task_tensors, weights):
    """
    Reshapes `weights` to match the shape of `task_tensors` by unsqeezing in the remaining dimenions.

    Args:
        task_tensors (`torch.Tensor`): The tensors that will be used to reshape `weights`.
        weights (`torch.Tensor`): The tensor to be reshaped.

    Returns:
        `torch.Tensor`: The reshaped tensor.
    """
    new_shape = weights.shape + (1,) * (task_tensors.dim() - weights.dim())
    weights = weights.view(new_shape)
    return weights


def magnitude_based_pruning(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """
    Prune the smallest values of the task tensors and retain the top-k values based on the specified fraction
    `density`.

    Args:
        tensor (`torch.Tensor`):The tensor to prune.
        density (`float`):The fraction of values to preserve. Should be in [0,1].

    Returns:
        `torch.Tensor`: The tensor with the pruned weights.
    """
    mask = torch.zeros_like(tensor).reshape(-1)
    k = int(density * tensor.numel())
    top_k = torch.topk(tensor.abs().reshape(-1), k=k, largest=True)
    mask[top_k[1]] = 1
    return tensor * mask.reshape(tensor.shape)


def random_pruning(tensor: torch.Tensor, density: float, rescale: bool) -> torch.Tensor:
    """
    Prune random values based on the specified fraction `density`.

    Args:
        tensor (`torch.Tensor`):The tensor to prune.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        rescale (`bool`):Whether to rescale the result to preserve the expected value of the original tensor.

    Returns:
        `torch.Tensor`: The pruned tensor.
    """
    mask = torch.bernoulli(torch.full_like(input=tensor, fill_value=density))
    pruned_tensor = tensor * mask
    if rescale:
        torch.div(input=pruned_tensor, other=density)
    return pruned_tensor


def prune(
    tensor: torch.Tensor,
    density: float,
    method: Literal["magnitude", "random"],
    rescale: bool = False,
) -> torch.Tensor:
    """
    Prune the values of task tensors based on the `method`.

    Args:
        tensor (`torch.Tensor`):The tensor to prune.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        method (`str`):The method to use to prune. Should be one of ["magnitude", "random"].
        rescale (`bool`):Whether to rescale the result to preserve the expected value of the original tensor.

    Returns:
        `torch.Tensor`: The pruned tensor.
    """
    if density >= 1:
        warnings.warn(
            f"The density {density} is greater than or equal to 1, no pruning will be performed."
        )
        return tensor
    elif density < 0:
        raise ValueError(f"Density should be >= 0, got {density}")
    if method == "magnitude":
        return magnitude_based_pruning(tensor, density)
    elif method == "random":
        return random_pruning(tensor, density, rescale=rescale)
    else:
        raise ValueError(f"Unknown method {method}")


def calculate_majority_sign_mask(
    tensor: torch.Tensor, method: Literal["total", "frequency"] = "total"
) -> torch.Tensor:
    """
    Get the mask of the majority sign across the task tensors. Task tensors are stacked on dimension 0.

    Args:
        tensor (`torch.Tensor`):The tensor to get the mask from.
        method (`str`):The method to use to get the mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The majority sign mask.
    """

    sign = tensor.sign()
    if method == "total":
        sign_magnitude = tensor.sum(dim=0)
    elif method == "frequency":
        sign_magnitude = sign.sum(dim=0)
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')
    majority_sign = torch.where(sign_magnitude >= 0, 1, -1)
    return sign == majority_sign


def disjoint_merge(
    task_tensors: torch.Tensor, majority_sign_mask: torch.Tensor
) -> torch.Tensor:
    """
    Merge the task tensors using disjoint merge.

    Args:
        task_tensors (`torch.Tensor`):The task tensors to merge.
        majority_sign_mask (`torch.Tensor`):The mask of the majority sign across the task tensors.

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    mixed_task_tensors = (task_tensors * majority_sign_mask).sum(dim=0)
    num_params_preserved = majority_sign_mask.sum(dim=0)
    return mixed_task_tensors / torch.clamp(num_params_preserved, min=1.0)


def task_arithmetic(
    task_tensors: list[torch.Tensor], weights: torch.Tensor
) -> torch.Tensor:
    """
    Merge the task tensors using `task arithmetic`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    task_tensors = torch.stack(task_tensors, dim=0)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def magnitude_prune(
    task_tensors: list[torch.Tensor], weights: torch.Tensor, density: float
) -> torch.Tensor:
    """
    Merge the task tensors using `task arithmetic`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`): The fraction of values to preserve. Should be in [0,1].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    # sparsify
    task_tensors = [
        prune(tensor, density, method="magnitude") for tensor in task_tensors
    ]
    task_tensors = torch.stack(task_tensors, dim=0)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def ties(
    task_tensors: list[torch.Tensor],
    weights: torch.Tensor,
    density: float,
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> torch.Tensor:
    """
    Merge the task tensors using `ties`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        majority_sign_method (`str`):
            The method to use to get the majority sign mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    # sparsify
    task_tensors = [
        prune(tensor, density, method="magnitude") for tensor in task_tensors
    ]
    task_tensors = torch.stack(task_tensors, dim=0)
    # Elect Sign
    majority_sign_mask = calculate_majority_sign_mask(
        task_tensors, method=majority_sign_method
    )
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    # Disjoint Merge
    mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
    return mixed_task_tensors


def dare_linear(
    task_tensors: list[torch.Tensor], weights: torch.Tensor, density: float
) -> torch.Tensor:
    """
    Merge the task tensors using `dare linear`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    # sparsify
    task_tensors = [
        prune(tensor, density, method="random", rescale=True) for tensor in task_tensors
    ]
    task_tensors = torch.stack(task_tensors, dim=0)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def dare_ties(
    task_tensors: list[torch.Tensor],
    weights: torch.Tensor,
    density: float,
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> torch.Tensor:
    """
    Merge the task tensors using `dare ties`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        majority_sign_method (`str`):
            The method to use to get the majority sign mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    # sparsify
    task_tensors = [
        prune(tensor, density, method="random", rescale=True) for tensor in task_tensors
    ]
    task_tensors = torch.stack(task_tensors, dim=0)
    # Elect Sign
    majority_sign_mask = calculate_majority_sign_mask(
        task_tensors, method=majority_sign_method
    )
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    # Disjoint Merge
    mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
    return mixed_task_tensors


def uniform_aggregate_stream(
    expert_file_path: str = "",
    expert_name_list: List[str] = [],
    aggregate_method: Literal["sum", "avg", "sign"] = "sum",
    prune_pct: float = None,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Aggregate LoRA expert weights to avoid loading all experts at once.

    Args:
        expert_file_path: Path to file containing expert names
        expert_name_list: List of expert names to process
        aggregate_method: Method for aggregating weights
            sum: sum the expert task vectors
            avg: average the expert task vectors
            sign: sum the sign of expert task vectors
        prune_pct: pruned task vector by top x % before aggregation
    Returns:
        Tuple of (aggregated_state_dict, processed_experts)
    """

    # Load expert list
    if len(expert_name_list) == 0 and expert_file_path != "":
        with open(expert_file_path, "r") as f:
            tmp = f.read()
        expert_name_list = tmp.split("\n")
        expert_name_list = [e for e in expert_name_list if e != ""]

    # Compute aggregate state dict
    agg_state_dict = {}
    processed_experts = []

    for i, expert in enumerate(expert_name_list):
        print(f"Processing adapter {i+1}/{len(expert_name_list)}: {expert}")

        try:
            expert_state_dict = get_normalized_weights(
                expert, apply_alpha=True, raise_error=True
            )
            lora_a_keys = [k for k in expert_state_dict.keys() if "lora_A" in k]

            for lora_a_key in lora_a_keys:
                lora_b_key = lora_a_key.replace("lora_A", "lora_B")
                parent_layer = lora_a_key.split(".lora_")[0]
                # Compute B @ A
                delta = torch.mm(
                    expert_state_dict[lora_b_key], expert_state_dict[lora_a_key]
                )
                # If weights should be pruned prior to aggregation
                if prune_pct:
                    delta = prune(delta, prune_pct, method="magnitude")
                # Accumulate
                if parent_layer not in agg_state_dict:
                    agg_state_dict[parent_layer] = torch.zeros_like(delta)

                if aggregate_method == "sign":
                    agg_state_dict[parent_layer].add_(delta.sign_())
                else:
                    agg_state_dict[parent_layer].add_(delta)
                del delta

            del expert_state_dict
            processed_experts.append(expert)

        except FileNotFoundError as e:
            print(f"{expert} not found: {e}")
        except RuntimeError as e:
            print(f"Expert {expert} failed to process (Runtime error): {e}")
        except Exception as e:
            print(f"Expert {expert} failed to process: {type(e).__name__}: {e}")

        if (i + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if aggregate_method == "avg":
        num_experts = len(processed_experts)
        if num_experts > 0:
            for key in agg_state_dict:
                agg_state_dict[key].div_(len(processed_experts))
        else:
            raise ValueError("No experts successfully processed")

    return agg_state_dict, processed_experts


def ties_stream(
    expert_file_path: str = "",
    expert_name_list: List[str] = [],
    expert_aggregate_state_dict_path: str = "",
    weight_init: Union[Literal["equal_int", "equal_frac"], float] = "equal_int",
    density: float = 0.2,
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Apply TIES merging method by loading expert one at a time.

    Args:
        expert_file_path: Path to file containing expert names
        expert_name_list: List of expert names
        expert_aggregate_state_dict_path: Path to pre-computed aggregated state
        density: Density parameter for pruning (0-1)
        majority_sign_method: Method for determining majority sign
    Returns:
        Tuple of (ties_output_dict, processed_experts)
    """

    # Validate density range
    if not 0 <= density <= 1:
        raise ValueError(f"Density must be between 0 and 1, got {density}")

    # Load the task expert list
    if len(expert_name_list) == 0 and expert_file_path != "":
        with open(expert_file_path, "r") as f:
            tmp = f.read()
        expert_name_list = tmp.split("\n")
        expert_name_list = [e for e in expert_name_list if e != ""]

    # Load aggregated experts to elect sign
    if expert_aggregate_state_dict_path:
        expert_aggregate_state_dict = torch.load(expert_aggregate_state_dict_path)
    else:
        if majority_sign_method == "total":
            aggregate_method = "sum"
        elif majority_sign_method == "frequency":
            aggregate_method = "sign"
        else:
            raise ValueError(f"Unknown majority sign method {majority_sign_method}")
        expert_aggregate_state_dict, _ = uniform_aggregate_stream(
            expert_file_path=expert_file_path,
            expert_name_list=expert_name_list,
            aggregate_method=aggregate_method,
            prune_pct=density,
        )
    # 1. Compute the majority sign
    sign_dict = {}
    for key in expert_aggregate_state_dict:
        signs = torch.where(
            expert_aggregate_state_dict[key] >= 0,
            torch.tensor(1, dtype=torch.int8),
            torch.tensor(-1, dtype=torch.int8),
        )
        sign_dict[key] = signs

    del expert_aggregate_state_dict

    # Loop over the experts to perform ties
    ties_output = {}
    ties_param_nonzero = {}
    processed_experts = []
    weights = []
    if weight_init == "equal_int":
        weights = [1] * len(expert_name_list)
    elif weight_init == "equal_frac":
        weights = [1 / len(expert_name_list)] * len(expert_name_list)
    elif isinstance(weight_init, float):
        weights = [weight_init] * len(expert_name_list)
    else:
        try:
            weight_init_value = float(weight_init)
            weights = [weight_init_value] * len(expert_name_list)
        except ValueError:
            pass

    print(f"weight initialized value: {weights}")

    for i, expert in enumerate(expert_name_list):
        print(f"Processing adapter {i+1}/{len(expert_name_list)}: {expert}")
        try:
            expert_state_dict = get_normalized_weights(expert, apply_alpha=True)
            lora_a_keys = [k for k in expert_state_dict.keys() if "lora_A" in k]

            for lora_a_key in lora_a_keys:
                lora_b_key = lora_a_key.replace("lora_A", "lora_B")
                parent_layer = lora_a_key.split(".lora_")[0]
                delta = torch.mm(
                    expert_state_dict[lora_b_key], expert_state_dict[lora_a_key]
                )

                # 2. Sparsify
                delta = prune(delta, density, method="magnitude")

                # 3. Apply sign mask
                maj_sign = sign_dict[parent_layer]
                if maj_sign.device != delta.device:
                    maj_sign = maj_sign.to(delta.device)

                maj_sign_mask = delta.sign() == maj_sign
                delta.mul_(maj_sign_mask)

                if parent_layer not in ties_output:
                    ties_output[parent_layer] = torch.zeros_like(delta)
                    ties_param_nonzero[parent_layer] = torch.zeros_like(delta)

                ties_output[parent_layer].add_(delta)
                if len(weights) > 0:
                    ties_output[parent_layer].mul_(weights[i])
                ties_param_nonzero[parent_layer].add_(
                    (maj_sign_mask != 0).to(delta.dtype)
                )

                del delta, maj_sign_mask
                if maj_sign.device != sign_dict[parent_layer].device:
                    del maj_sign

            del expert_state_dict
            processed_experts.append(expert)

        except FileNotFoundError as e:
            print(f"Expert {expert} not found: {e}")
        except RuntimeError as e:
            print(f"Expert {expert} failed to process (Runtime error): {e}")
        except Exception as e:
            print(f"Expert {expert} failed to process: {type(e).__name__}: {e}")

        if (i + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 4. Average the ties_output by num. nonzero params
    if len(processed_experts) == 0:
        raise ValueError("No experts successfully processed")

    for key in ties_output:
        ties_output[key].div_(torch.clamp(ties_param_nonzero[key], min=1))

    del ties_param_nonzero

    return ties_output, processed_experts


def tsv_merge(
    task_tensors: list[torch.Tensor | None], weights: list[float], reduced_size: int
) -> torch.Tensor:
    """
    Adapted from https://github.com/AntoAndGar/task_singular_vectors/blob/main/src/utils/TSVM_utils.py#L115 compute_and_sum_svd_mem_reduction_lossless

    Args:
        task_tensors (list[torch.Tensor]): The list of task tensors.
        reduced_size (int): The reduced size of each tensor.

    Returns:
        torch.Tensor: The merged tensor.
    """

    assert len(task_tensors) > 0, "No task tensors provided for merging."
    assert all(
        task_tensors[0].shape == t.shape for t in task_tensors
    ), "All task tensors must have the same shape."
    assert task_tensors[0].dim() == 2, "All task tensors must be 2D."
    assert len(task_tensors) == len(
        weights
    ), "Task tensors and weights must have the same length."

    first_valid_idx = min([i for i, t in enumerate(task_tensors) if t is not None])
    device = task_tensors[first_valid_idx].device
    num_tasks = len(task_tensors)

    rows, cols = task_tensors[first_valid_idx].shape

    # Initialize containers for stacked SVD components
    # Assuming reduced_size is per model, so total columns is num_tasks * reduced_size
    sum_u = torch.zeros(rows, reduced_size * num_tasks, device=device)
    sum_s = torch.zeros(reduced_size * num_tasks, device=device)
    sum_v = torch.zeros(reduced_size * num_tasks, cols, device=device)

    for i, tensor in enumerate(task_tensors):
        if tensor is None:
            continue
        # Ensure float for SVD
        u, s, v = torch.linalg.svd(tensor.float(), full_matrices=False)

        # Truncate to reduced_size
        current_k = min(reduced_size, s.shape[0])

        start_idx = i * reduced_size
        end_idx = start_idx + current_k

        # Place components
        sum_u[:, start_idx:end_idx] = u[:, :current_k]
        sum_s[start_idx:end_idx] = s[:current_k] * weights[i]
        # v from torch.linalg is Vh (transposed), so shape is (min(r,c), c)
        sum_v[start_idx:end_idx, :] = v[:current_k, :]

    # Compute SVD of the stacked components
    u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
    u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)

    merged_tensor = torch.linalg.multi_dot(
        (
            u_u,
            v_u,
            torch.diag(sum_s),
            u_v,
            v_v,
        )
    )

    return merged_tensor.to(task_tensors[0].dtype)


def tsv_merge_stream(
    all_subset_layers: List[List],
    expert_file_path: str = "",
    expert_name_list: List[str] = [],
    weight_init: Literal["equal_int", "equal_frac"] = "equal_int",
    reduced_size: int = 16,
) -> Dict[str, torch.Tensor]:
    """
    TSV merging for a large number of experts that do not fit in one device.
    The function performs layer-module-wise merging of LoRA adapters from multiple expert models.

    Args:
        all_subset_layers: List of layer subsets to process.
                           Each subset specifies layers to load.
        expert_file_path (str, optional): Path to a text file containing expert names.
        expert_name_list (List[str], optional): List of expert model names/paths to merge.
        reduced_size (int, optional): Number of the original rank to retain after SVD reduction.

    Returns:
        torch.Tensor: The merged tensor for the whole base model.
    """

    # Load the task expert list
    if len(expert_name_list) == 0 and expert_file_path != "":
        with open(expert_file_path, "r") as f:
            tmp = f.read()
        expert_name_list = tmp.split("\n")
        expert_name_list = [e for e in expert_name_list if e != ""]

    merged_weights = {}
    num_experts = len(expert_name_list)

    weights = []
    if weight_init == "equal_int":
        weights = [1] * len(expert_name_list)
    elif weight_init == "equal_frac":
        weights = [1 / len(expert_name_list)] * len(expert_name_list)
    else:
        pass

    # Load specific expert layers in memory
    for subset_layers in all_subset_layers:
        accumulated_sum_u = {}
        accumulated_sum_s = {}
        accumulated_sum_v = {}

        for idx, expert in enumerate(expert_name_list):
            print(f"Loading {expert}...")
            expert_state_dict = get_normalized_weights(
                expert, apply_alpha=True, subset_layers=subset_layers
            )
            lora_a_keys = [k for k in expert_state_dict.keys() if "lora_A" in k]

            for lora_a_key in lora_a_keys:
                lora_b_key = lora_a_key.replace("lora_A", "lora_B")
                parent_layer = lora_a_key.split(".lora_")[0]
                delta = torch.mm(
                    expert_state_dict[lora_b_key], expert_state_dict[lora_a_key]
                )

                if parent_layer not in accumulated_sum_u:
                    device = delta.device
                    rows, cols = delta.shape
                    accumulated_sum_u[parent_layer] = torch.zeros(
                        rows, reduced_size * num_experts, device=device
                    )
                    accumulated_sum_s[parent_layer] = torch.zeros(
                        reduced_size * num_experts, device=device
                    )
                    accumulated_sum_v[parent_layer] = torch.zeros(
                        reduced_size * num_experts, cols, device=device
                    )

                u, s, v = torch.linalg.svd(delta.float(), full_matrices=False)
                current_k = min(reduced_size, s.shape[0])
                start_idx = idx * reduced_size
                end_idx = start_idx + current_k

                accumulated_sum_u[parent_layer][:, start_idx:end_idx].copy_(
                    u[:, :current_k]
                )
                accumulated_sum_s[parent_layer][start_idx:end_idx].copy_(s[:current_k])
                if len(weights) > 0:
                    accumulated_sum_s[parent_layer][start_idx:end_idx].mul_(
                        weights[idx]
                    )
                accumulated_sum_v[parent_layer][start_idx:end_idx, :].copy_(
                    v[:current_k, :]
                )

                del delta, u, s, v
            del expert_state_dict

            # gc.collect()
            # if torch.cuda.is_available():
            #    torch.cuda.empty_cache()

        # final SVD after iterating through all experts on the given layers
        parent_layers = list(accumulated_sum_u.keys())
        for parent_layer in parent_layers:
            u_u, _, v_u = torch.linalg.svd(
                accumulated_sum_u[parent_layer], full_matrices=False
            )
            u_v, _, v_v = torch.linalg.svd(
                accumulated_sum_v[parent_layer], full_matrices=False
            )

            merged_weights[parent_layer] = torch.linalg.multi_dot(
                (u_u, v_u, torch.diag(accumulated_sum_s[parent_layer]), u_v, v_v)
            )
            del u_u, v_u, u_v, v_v
            del accumulated_sum_u[parent_layer]
            del accumulated_sum_s[parent_layer]
            del accumulated_sum_v[parent_layer]

        del accumulated_sum_u, accumulated_sum_s, accumulated_sum_v

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return merged_weights


def core_space_merge(
    lora_A_tensors: list[torch.Tensor | None],
    lora_B_tensors: list[torch.Tensor | None],
    lora_configs: list[LoraConfig],
    weights: list[float],
) -> torch.Tensor:
    """
    Merge LoRA A and B tensors using the core space method.

    Args:
        lora_A_tensors (list[torch.Tensor (r_i, in_dim) | None]): The list of lora A tensors. None if this task doesn't have LoRA for the current module.
        lora_B_tensors (list[torch.Tensor (out_dim, r_i) | None]): The list of lora B tensors. None if this task doesn't have LoRA for the current module.
        weights (list[float]): The list of weights.

    Returns:
        torch.Tensor: The merged tensor.
    """
    assert len(lora_A_tensors) == len(
        lora_B_tensors
    ), "lora_A_tensors and lora_B_tensors must have the same length."
    assert len(lora_A_tensors) == len(
        weights
    ), "lora_A_tensors and weights must have the same length."
    assert len(lora_A_tensors) == len(
        lora_configs
    ), "lora_A_tensors and lora_configs must have the same length."

    valid_indices = [
        i
        for i, (A, B) in enumerate(zip(lora_A_tensors, lora_B_tensors))
        if A is not None and B is not None
    ]
    if not valid_indices:
        raise ValueError("All LoRA tensors are None.")

    if len(valid_indices) != len(
        [tensor for tensor in lora_A_tensors if tensor is not None]
    ) or len(valid_indices) != len(
        [tensor for tensor in lora_B_tensors if tensor is not None]
    ):
        raise ValueError("None tensors are not aligned.")

    first_idx = valid_indices[0]
    device = lora_A_tensors[first_idx].device
    dtype = lora_A_tensors[first_idx].dtype

    # Stack lora tensors
    valid_Bs = [lora_B_tensors[i] for i in valid_indices]
    B_stack = torch.cat(valid_Bs, dim=1)  # (out, sum(r_i))

    valid_As_T = [lora_A_tensors[i].t() for i in valid_indices]
    A_stack = torch.cat(valid_As_T, dim=1)  # (in, sum(r_i))

    # Compute shared bases U (from B) and V (from A)
    U_B, _, _ = torch.linalg.svd(B_stack.to(torch.float64), full_matrices=False)
    U = U_B.to(device=device, dtype=dtype)
    k_u = U.shape[1]

    U_A, _, _ = torch.linalg.svd(A_stack.to(torch.float64), full_matrices=False)
    V = U_A.to(device=device, dtype=dtype)
    k_v = V.shape[1]

    # Compute Core Matrices (Sigmas)
    merged_sigma = torch.zeros(k_u, k_v, device=device, dtype=dtype)

    for i in valid_indices:
        # Calculate sigma for this task
        # s_t = U^T @ B_i @ A_i @ V
        term1 = U.t() @ lora_B_tensors[i]  # (k_u, out) @ (out, r_i) -> (k_u, r_i)
        term2 = lora_A_tensors[i] @ V  # (r_i, in) @ (in, k_v) -> (r_i, k_v)
        sigma_t = term1 @ term2  # (k_u, k_v)

        # Scaling factor from LoRA config
        config = lora_configs[i]
        scaling = config.lora_alpha / (config.r**0.5 if config.use_rslora else config.r)

        # Weighted sum with scaling
        merged_sigma += sigma_t * weights[i] * scaling

    # Reconstruct Merged Weight
    # W_merge = U @ Sigma_merge @ V^T
    merged_weight = U @ merged_sigma @ V.t()

    return merged_weight
