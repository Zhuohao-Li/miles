from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass

import torch

from miles.backends.training_utils.cp_utils import get_logits_and_tokens_offset_with_cp
from miles.backends.training_utils.loss_hub.math_utils import (
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import distributed_masked_whiten


@dataclass(frozen=True)
class AdvantageEstimatorInput:
    args: Namespace
    kl: list[torch.Tensor]
    rewards: list[float]
    log_probs: list[torch.Tensor] | None
    loss_masks: list[torch.Tensor]
    total_lengths: list[int]
    response_lengths: list[int]
    values: list[torch.Tensor] | None
    max_seq_lens: list[int] | None


AdvantageEstimator = Callable[
    [AdvantageEstimatorInput],
    tuple[list[torch.Tensor], list[torch.Tensor]],
]
_ADVANTAGE_ESTIMATORS: dict[str, AdvantageEstimator] = {}


def register_advantage_estimator(name: str) -> Callable[[AdvantageEstimator], AdvantageEstimator]:
    """Register an advantage estimator under its configuration name."""

    def register(estimator: AdvantageEstimator) -> AdvantageEstimator:
        if name in _ADVANTAGE_ESTIMATORS:
            raise ValueError(f"Advantage estimator {name!r} is already registered")
        _ADVANTAGE_ESTIMATORS[name] = estimator
        return estimator

    return register


def get_advantage_estimator(name: str) -> AdvantageEstimator:
    """Return a registered estimator or reject an unsupported name."""

    try:
        return _ADVANTAGE_ESTIMATORS[name]
    except KeyError as exc:
        raise NotImplementedError(f"advantage_estimator {name} is not supported. ") from exc


@register_advantage_estimator("grpo")
@register_advantage_estimator("gspo")
def _compute_grpo(inputs: AdvantageEstimatorInput) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    rewards = torch.tensor(inputs.rewards, dtype=torch.float32, device=inputs.kl[0].device)
    returns = get_grpo_returns(rewards, inputs.kl)
    return list(returns), returns


@register_advantage_estimator("ppo")
def _compute_ppo(inputs: AdvantageEstimatorInput) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    token_rewards = []
    kl_coef = -inputs.args.kl_coef
    for kl in inputs.kl:
        kl *= kl_coef
        token_rewards.append(kl)
    return get_advantages_and_returns_batch(
        total_lengths=inputs.total_lengths,
        response_lengths=inputs.response_lengths,
        values_list=inputs.values,
        rewards_list=token_rewards,
        terminal_rewards=inputs.rewards,
        qkv_format=inputs.args.qkv_format,
        max_seq_lens=inputs.max_seq_lens,
        loss_masks=inputs.loss_masks,
        gamma=inputs.args.gamma,
        lambd=inputs.args.lambd,
    )


@register_advantage_estimator("reinforce_plus_plus")
def _compute_reinforce_plus_plus(
    inputs: AdvantageEstimatorInput,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    rewards = torch.tensor(inputs.rewards, dtype=torch.float32, device=inputs.kl[0].device)
    returns = get_reinforce_plus_plus_returns(
        rewards=rewards,
        kl=inputs.kl,
        loss_masks=inputs.loss_masks,
        response_lengths=inputs.response_lengths,
        total_lengths=inputs.total_lengths,
        kl_coef=inputs.args.kl_coef,
        gamma=inputs.args.gamma,
    )
    return list(returns), returns


@register_advantage_estimator("reinforce_plus_plus_baseline")
def _compute_reinforce_plus_plus_baseline(
    inputs: AdvantageEstimatorInput,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    rewards = torch.tensor(inputs.rewards, dtype=torch.float32, device=inputs.kl[0].device)
    advantages = get_reinforce_plus_plus_baseline_advantages(
        rewards=rewards,
        kl=inputs.kl,
        loss_masks=inputs.loss_masks,
        kl_coef=inputs.args.kl_coef,
    )
    return advantages, advantages


def compute_advantages(
    args: Namespace,
    kl: list[torch.Tensor],
    rewards: list[float],
    log_probs: list[torch.Tensor] | None,
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    values: list[torch.Tensor] | None = None,
    max_seq_lens: list[int] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Dispatch to the configured advantage estimator.

    Shape symbols:
        `B`: Number of samples in the current local batch.
        `T_i`: Prompt-plus-response length of sample `i`, excluding BSHD padding.
        `R_i`: Full response length of sample `i` before CP splitting.
        `C_i`: Number of response-aligned positions of sample `i` stored on this CP rank; prompt and padding positions are excluded.
        `P_i`: Padded sequence length of sample `i` used by BSHD CP splitting.

    Args:
        args: `Namespace`; no tensor shape.
        kl: List length `B`; `kl[i]` has shape `[C_i]`.
        rewards: List length `B`; `rewards[i]` is a scalar.
        log_probs: `None` or list length `B`; `log_probs[i]` has shape `[C_i]`.
        loss_masks: List length `B`; `loss_masks[i]` has shape `[R_i]`.
        total_lengths: List length `B`; `total_lengths[i] = T_i`.
        response_lengths: List length `B`; `response_lengths[i] = R_i`.
        values: `None` or list length `B`; `values[i]` has shape `[C_i]`. PPO requires this input.
        max_seq_lens: `None` or list length `B`; `max_seq_lens[i] = P_i`. Required for BSHD with CP.

    `C_i = R_i` when CP size is 1. With CP size greater than 1, `0 <= C_i <= R_i`; `C_i` can be zero and can differ across ranks. THD partitions a sequence of length `T_i`, while BSHD partitions the padded maximum sequence length, so the two formats do not guarantee the same `C_i`.

    Returns:
        `advantages`: List length `B`; `advantages[i]` has shape `[C_i]`.
        `returns`: List length `B`; `returns[i]` has shape `[C_i]`.
    """
    estimator = get_advantage_estimator(args.advantage_estimator)
    return estimator(
        AdvantageEstimatorInput(
            args=args,
            kl=kl,
            rewards=rewards,
            log_probs=log_probs,
            loss_masks=loss_masks,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            values=values,
            max_seq_lens=max_seq_lens,
        )
    )


def normalize_advantages(
    args: Namespace,
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> list[torch.Tensor]:
    """Whiten advantages across the DP group using `loss_masks` for weighting.

    Under CP > 1 the mask is sliced to this rank's tokens; when the local
    mask is empty the inputs pass through unchanged. Output shapes match
    `advantages`.
    """
    num_samples = len(advantages)
    assert len(loss_masks) == num_samples
    assert len(total_lengths) == num_samples
    assert len(response_lengths) == num_samples
    if max_seq_lens is not None:
        assert len(max_seq_lens) == num_samples

    parallel_state = get_parallel_state()
    all_advs = torch.cat(advantages)
    cp_size = parallel_state.cp.size
    if cp_size == 1:
        all_masks = torch.cat(loss_masks)
    else:
        mask_chunks = []
        max_seq_lens_iter = max_seq_lens if max_seq_lens is not None else [None] * num_samples
        for total_len, response_len, full_mask, max_seq_len in zip(
            total_lengths, response_lengths, loss_masks, max_seq_lens_iter, strict=True
        ):
            prompt_len = total_len - response_len

            _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                total_len, response_len, args.qkv_format, max_seq_len
            )

            # Convert global offsets to response-space offsets
            (s0, e0), (s1, e1) = token_offsets
            res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
            res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

            local_mask_parts = []
            if res_e0 > res_s0:
                local_mask_parts.append(full_mask[res_s0:res_e0])
            if res_e1 > res_s1:
                local_mask_parts.append(full_mask[res_s1:res_e1])

            # Concatenate the parts to form the final mask chunk for this rank and this sequence
            local_mask_chunk = (
                torch.cat(local_mask_parts)
                if local_mask_parts
                else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
            )
            mask_chunks.append(local_mask_chunk)

        all_masks = torch.cat(mask_chunks)

    if all_masks.numel() > 0:
        assert (
            all_advs.size() == all_masks.size()
        ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
        dp_group = parallel_state.effective_dp.group

        whitened_advs_flat = distributed_masked_whiten(
            all_advs,
            all_masks,
            process_group=dp_group,
            shift_mean=True,
        )
        chunk_lengths = [chunk.size(0) for chunk in advantages]
        advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    return advantages
