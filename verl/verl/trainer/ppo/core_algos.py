# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ['register', "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

## 新增
from typing import Dict, Tuple
ADV_ESTIMATOR_REGISTRY = {}

def register_adv_est(name_or_enum):
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """
    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}")
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn
    return decorator

def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]

class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    ## 新增
    GROUP = "group_ppo"  # 用于GROUP_PPO的advantage计算


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError

@register_adv_est(AdvantageEstimator.GAE) # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO) # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.GRPO_PASSK) # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config = None,
    **kwargs,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (dict) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages

@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE) # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor,
                                                           epsilon: float = 1e-6, config=None, **kwargs):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.RLOO) # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray,
                                   epsilon: float = 1e-6, config=None, **kwargs):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.OPO) # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6,
                                  config=None, **kwargs):
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.tensor(id2score[idx])
                len_tensor = torch.tensor(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS) # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config=None, **kwargs):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns

@register_adv_est(AdvantageEstimator.REMAX) # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor, config=None, **kwargs):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    group_endpoints=None,  # 新增参数，用于GROUP算法
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    # ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    
    ## 新增
    # 对于GROUP算法，需要先应用endpoint mask
    if group_endpoints is not None:
        endpoint_mask = group_endpoints.float() * response_mask
        effective_mask = endpoint_mask
    else:
        effective_mask = response_mask
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, effective_mask)
    ## end新增
    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    
    # pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    ## 新增
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), effective_mask)
    ## end新增
    
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    
    # pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)
    ## 新增
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), effective_mask)
    ## end新增
    
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    
    # pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    ## 新增
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=effective_mask, loss_agg_mode=loss_agg_mode)
    ## end新增
    
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data

## 新增
# ### 计算GROUP_PPO的advantage 
# @register_adv_est(AdvantageEstimator.GROUP)
# def compute_group_advantage(
#     token_level_rewards: torch.Tensor,
#     response_mask: torch.Tensor,
#     index: np.ndarray,
#     values: torch.Tensor, 
#     config=None,
#     **kwargs,
# ):
#     """
#     Compute advantage for GROUP algorithm based on dynamic GRPO vs PPO variance comparison.
    
#     This implementation follows Algorithm 2 Group from the paper, which dynamically
#     decides grouping strategy based on the ratio of GRPO variance to PPO variance.
    
#     Args:
#         token_level_rewards: (batch_size, response_length)
#         response_mask: (batch_size, response_length) 
#         index: (batch_size,) → group ID per sample
#         config: Configuration containing GROUP algorithm parameters
        
#     Returns:
#         advantages: (batch_size, response_length)
#         returns: (batch_size, response_length)
#     """
#     assert config is not None, "GROUP algorithm requires config parameters"
#     assert values is not None, "GROUP algorithm requires critic values"

#     batch_size, response_length = token_level_rewards.shape
#     scores = token_level_rewards.sum(dim=-1)  # 总奖励分数
    
#     # 初始化advantage为零
#     advantages = torch.zeros_like(token_level_rewards)
#     returns = torch.zeros_like(token_level_rewards)
    
#     # 按组分组
#     id2indices = defaultdict(list)
#     id2scores = defaultdict(list)
#     id2values = defaultdict(list)
#     id2masks = defaultdict(list)
    
#     for i in range(batch_size):
#         group_id = index[i]
#         id2indices[group_id].append(i)
#         id2scores[group_id].append(scores[i])
#         id2values[group_id].append(values[i])  # 使用critic values
#         id2masks[group_id].append(response_mask[i])
    
#     with torch.no_grad():
#         for group_id, group_indices in id2indices.items():
#             group_scores = torch.stack(id2scores[group_id])
#             group_values = torch.stack(id2values[group_id])  # critic values
#             group_masks = torch.stack(id2masks[group_id])
#             group_size = len(group_indices)
            
#             if group_size < 2:
#                 # 单样本组，直接计算
#                 single_idx = group_indices[0]
#                 advantage_val = group_scores[0] - group_values[0].sum()  # reward - value
#                 advantages[single_idx] = advantage_val * response_mask[single_idx]
#                 returns[single_idx] = advantages[single_idx] + group_values[0]
#                 continue
            
#             # 使用Algorithm 2确定组边界
#             r_ppo = 0.0
#             r_grpo = 0.0
#             group_boundaries = []  # 记录组边界位置
            
#             # Compute group mean for baseline
#             group_mean_reward = torch.mean(group_scores)
#             group_mean_value = torch.mean(group_values.sum(dim=-1))
            
#             for n in range(group_size):
#                 # 计算当前位置的梯度范数平方（简化为advantage平方）
#                 current_advantage = group_scores[n] - group_values[n].sum()
#                 current_grad_norm_sq = current_advantage ** 2
                
#                 # 累积PPO方差 (Algorithm 2: r_ppo += ||∇π_θ(s_n)||² Â_t²)
#                 r_ppo += current_grad_norm_sq.item()
                
#                 # 计算GRPO方差 (Algorithm 2: r_grpo = ||∇(∏ π_θ(s_n))||² Â_t²)
#                 if n > 0:
#                     # 累积策略梯度范数（简化实现）
#                     cumulative_advantages = group_scores[:n+1] - group_mean_value
#                     cumulative_grad_norm_sq = torch.mean(cumulative_advantages ** 2)
#                     r_grpo = cumulative_grad_norm_sq.item() * (n + 1)
#                 else:
#                     r_grpo = current_grad_norm_sq.item()
                
#                 # 检查分组条件：n ≥ r_grpo/r_ppo
#                 if r_ppo > 1e-8 and n >= (r_grpo / r_ppo):
#                     group_boundaries.append(n)  # 记录边界
#                     r_ppo = 0.0
#                     r_grpo = 0.0
            
#             # 确保最后一个位置也是边界
#             if group_size - 1 not in group_boundaries:
#                 group_boundaries.append(group_size - 1)
            
#             print(f"组 {group_id}: 大小={group_size}, 边界位置={group_boundaries}")
    
#             subgroup_start = 0
#             for boundary_pos in group_boundaries:
#                 # 计算这个子组的advantage (只在末端)
#                 subgroup_indices = list(range(subgroup_start, boundary_pos + 1))
#                 subgroup_scores = group_scores[subgroup_indices]
#                 subgroup_values = group_values[subgroup_indices]
                
#                 # 子组baseline (使用critic values)
#                 subgroup_value_baseline = torch.mean(subgroup_values.sum(dim=-1))
#                 subgroup_reward_mean = torch.mean(subgroup_scores)
                
#                 # 🎯 关键：只为边界位置（末端）计算advantage
#                 endpoint_idx = group_indices[boundary_pos]  # 全局索引
#                 endpoint_advantage = subgroup_reward_mean - subgroup_value_baseline
                
#                 # 只有末端样本有advantage，其他样本advantage为0
#                 advantages[endpoint_idx] = endpoint_advantage * response_mask[endpoint_idx]
#                 returns[endpoint_idx] = advantages[endpoint_idx] + group_values[boundary_pos]
                
#                 # 子组内其他样本的advantage保持为0（已经初始化为0）
#                 for sub_idx in subgroup_indices[:-1]:  # 除了末端
#                     global_idx = group_indices[sub_idx]
#                     advantages[global_idx] = 0.0  # 明确设置为0
#                     returns[global_idx] = group_values[sub_idx]  # 只有value，没有advantage
                
#                 print(f"  子组 [{subgroup_start}:{boundary_pos}]: 只在位置{boundary_pos}计算advantage={endpoint_advantage:.4f}")
                
#                 subgroup_start = boundary_pos + 1
    
#     print(f"✅ 末端计算完成: {torch.sum(advantages != 0).item()} 个样本有非零advantage")
    
#     return advantages, returns


# ## 计算GROUP_PPO的方差指标，方便检测和调参
# def compute_group_variance_metrics(
#     token_level_rewards: torch.Tensor,
#     response_mask: torch.Tensor, 
#     index: np.ndarray,
#     advantages: torch.Tensor,
# ) -> Dict[str, float]:
#     """
#     Compute variance metrics for GROUP algorithm monitoring.
    
#     Args:
#         token_level_rewards: Token-level reward scores
#         response_mask: Response attention mask
#         index: Group indices 
#         advantages: Computed advantages
        
#     Returns:
#         Dictionary of variance metrics
#     """
#     batch_size = token_level_rewards.shape[0]
#     scores = token_level_rewards.sum(dim=-1)
    
#     # Group-wise variance computation
#     id2scores = defaultdict(list)
#     for i in range(batch_size):
#         id2scores[index[i]].append(scores[i])
    
#     group_variances = []
#     ppo_variances = []
#     grpo_variances = []
    
#     for group_scores in id2scores.values():
#         if len(group_scores) > 1:
#             group_scores_tensor = torch.tensor(group_scores)
#             group_mean = torch.mean(group_scores_tensor)
            
#             # PPO-style variance (individual)
#             ppo_var = torch.var(group_scores_tensor).item()
#             ppo_variances.append(ppo_var)
            
#             # GRPO-style variance (cumulative)
#             cumulative_vars = []
#             for i in range(len(group_scores)):
#                 cum_scores = group_scores_tensor[:i+1]
#                 cum_var = torch.var(cum_scores).item() if len(cum_scores) > 1 else 0.0
#                 cumulative_vars.append(cum_var)
#             grpo_var = np.mean(cumulative_vars)
#             grpo_variances.append(grpo_var)
            
#             group_variances.append(ppo_var)
    
#     metrics = {
#         "group/ppo_variance_mean": np.mean(ppo_variances) if ppo_variances else 0.0,
#         "group/grpo_variance_mean": np.mean(grpo_variances) if grpo_variances else 0.0,
#         "group/variance_ratio_mean": np.mean([g/(p+1e-8) for g, p in zip(grpo_variances, ppo_variances)]) if grpo_variances and ppo_variances else 0.0,
#         "group/num_groups": len(group_variances),
#         "group/avg_group_size": batch_size / len(group_variances) if group_variances else 1.0,
#     }
    
#     return metrics
#     ## end新增

## 新增：Token-Level分组的核心实现
# @register_adv_est(AdvantageEstimator.GROUP)
# def compute_group_advantage(
#     token_level_rewards: torch.Tensor,
#     response_mask: torch.Tensor,
#     index: np.ndarray,
#     values: torch.Tensor, 
#     config=None,
#     **kwargs,
# ):
#     """
#     实现真正的token级别GROUP算法，基于Algorithm 2
#     只在每个组的端点token计算advantage
#     """
#     batch_size, response_length = token_level_rewards.shape
    
#     # 初始化所有位置的advantage为零
#     advantages = torch.zeros_like(token_level_rewards)
#     returns = torch.zeros_like(token_level_rewards)
    
#     # 按组分组处理
#     id2indices = defaultdict(list)
#     for i in range(batch_size):
#         id2indices[index[i]].append(i)
    
#     with torch.no_grad():
#         for group_id, group_indices in id2indices.items():
#             if len(group_indices) < 2:
#                 # 单样本组，标准处理
#                 single_idx = group_indices[0]
#                 single_adv = token_level_rewards[single_idx] - values[single_idx]
#                 advantages[single_idx] = single_adv * response_mask[single_idx]
#                 returns[single_idx] = advantages[single_idx] + values[single_idx]
#                 continue
            
#             # 对组内每个序列实现Algorithm 2的逻辑
#             for seq_idx_in_group, global_seq_idx in enumerate(group_indices):
#                 seq_rewards = token_level_rewards[global_seq_idx]
#                 seq_values = values[global_seq_idx]
#                 seq_mask = response_mask[global_seq_idx]
                
#                 # Algorithm 2: 动态确定端点
#                 r_ppo = 0.0
#                 r_grpo = 0.0
#                 endpoints = []
                
#                 for t in range(response_length):
#                     if seq_mask[t] == 0:
#                         continue
                    
#                     # 计算梯度范数平方
#                     current_advantage = seq_rewards[t] - seq_values[t]
#                     grad_norm_sq = current_advantage ** 2
                    
#                     # 累积PPO方差
#                     r_ppo += grad_norm_sq.item()
                    
#                     # 计算GRPO方差 
#                     if t > 0:
#                         cum_advantages = seq_rewards[:t+1] - seq_values[:t+1]
#                         cum_advantages = cum_advantages * seq_mask[:t+1]
#                         r_grpo = torch.mean(cum_advantages ** 2).item() * (t + 1)
#                     else:
#                         r_grpo = grad_norm_sq.item()
                    
#                     # Algorithm 2检查条件: t ≥ r_grpo/r_ppo
#                     if r_ppo > 1e-8 and t >= (r_grpo / r_ppo):
#                         endpoints.append(t)
#                         r_ppo = 0.0
#                         r_grpo = 0.0
                
#                 # 确保最后一个有效token是端点
#                 valid_positions = torch.where(seq_mask > 0)[0]
#                 if len(valid_positions) > 0:
#                     last_pos = valid_positions[-1].item()
#                     if last_pos not in endpoints:
#                         endpoints.append(last_pos)
                
#                 # 只在端点计算advantage
#                 for endpoint_pos in endpoints:
#                     if endpoint_pos < response_length and seq_mask[endpoint_pos] > 0:
#                         endpoint_reward = seq_rewards[endpoint_pos]
#                         endpoint_value = seq_values[endpoint_pos]
                        
#                         # 使用组内对比作为baseline
#                         group_baseline = torch.mean(torch.stack([
#                             token_level_rewards[other_idx, endpoint_pos] 
#                             for other_idx in group_indices 
#                             if other_idx != global_seq_idx and endpoint_pos < response_length
#                             and response_mask[other_idx, endpoint_pos] > 0
#                         ])) if len(group_indices) > 1 else endpoint_value
                        
#                         endpoint_advantage = endpoint_reward - group_baseline
#                         advantages[global_seq_idx, endpoint_pos] = endpoint_advantage
#                         returns[global_seq_idx, endpoint_pos] = endpoint_advantage + endpoint_value
    
#     return advantages, returns

# def compute_token_group_metrics(
#     token_level_rewards: torch.Tensor,
#     response_mask: torch.Tensor,
#     advantages: torch.Tensor,
#     values: torch.Tensor,
#     config=None,
# ) -> Dict[str, float]:
#     """
#     计算Token-Level分组的监控指标
    
#     Args:
#         token_level_rewards: Token级别的reward
#         response_mask: Response的有效mask
#         advantages: 计算得到的advantage
#         values: Critic的value估计
#         config: 算法配置
        
#     Returns:
#         包含各种监控指标的字典
#     """
#     batch_size, response_length = token_level_rewards.shape
    
#     # 统计指标
#     total_tokens = int(response_mask.sum().item())
#     non_zero_advantages = int((advantages != 0.0).sum().item())
    
#     if total_tokens == 0:
#         return {
#             "token_group/total_tokens": 0.0,
#             "token_group/advantage_positions": 0.0,
#             "token_group/compression_ratio": 0.0,
#             "token_group/avg_group_size": 0.0,
#         }
    
#     compression_ratio = 1.0 - (non_zero_advantages / total_tokens)
#     avg_group_size = total_tokens / max(non_zero_advantages, 1)
    
#     # 计算方差指标
#     valid_rewards = token_level_rewards[response_mask.bool()]
#     valid_values = values[response_mask.bool()]
#     valid_advantages = advantages[response_mask.bool()]
    
#     reward_variance = torch.var(valid_rewards).item() if len(valid_rewards) > 1 else 0.0
#     advantage_variance = torch.var(valid_advantages).item() if len(valid_advantages) > 1 else 0.0
    
#     metrics = {
#         "token_group/total_tokens": float(total_tokens),
#         "token_group/advantage_positions": float(non_zero_advantages),
#         "token_group/compression_ratio": compression_ratio,
#         "token_group/avg_group_size": avg_group_size,
#         "token_group/reward_variance": reward_variance,
#         "token_group/advantage_variance": advantage_variance,
#         "token_group/speedup_factor": 1.0 / (1.0 - compression_ratio + 1e-8),
#     }
    
#     return metrics


## v_1 无监控版本
# @register_adv_est(AdvantageEstimator.GROUP)
# def compute_group_advantage(
#     token_level_rewards: torch.Tensor,
#     response_mask: torch.Tensor,
#     index: np.ndarray,
#     values: torch.Tensor, 
#     old_log_prob: torch.Tensor,  # 需要添加这个参数
#     log_prob: torch.Tensor,      # 需要添加这个参数
#     config=None,
#     **kwargs,
# ):
#     """
#     实现Group-based PPO算法
#     Algorithm 1: 主PPO训练流程，只在组端点计算advantage
#     Algorithm 2: 动态分组判断
#     """
#     batch_size, response_length = token_level_rewards.shape
    
#     # 初始化
#     advantages = torch.zeros_like(token_level_rewards)
#     returns = torch.zeros_like(token_level_rewards)
    
#     # 按组处理
#     id2indices = defaultdict(list)
#     for i in range(batch_size):
#         id2indices[index[i]].append(i)
    
#     with torch.no_grad():
#         for group_id, group_indices in id2indices.items():
#             for global_seq_idx in group_indices:
#                 seq_rewards = token_level_rewards[global_seq_idx]
#                 seq_values = values[global_seq_idx]
#                 seq_mask = response_mask[global_seq_idx]
#                 seq_old_logprob = old_log_prob[global_seq_idx]
#                 seq_new_logprob = log_prob[global_seq_idx]
                
#                 # Algorithm 2: 动态确定分组端点
#                 endpoints = _find_group_endpoints(
#                     seq_mask, seq_old_logprob, seq_new_logprob, 
#                     seq_rewards, seq_values
#                 )
                
#                 # Algorithm 1: 只在端点计算advantage
#                 for endpoint_pos in endpoints:
#                     if endpoint_pos < response_length and seq_mask[endpoint_pos] > 0:
#                         # 计算到端点的累积奖励
#                         cumulative_reward = torch.sum(
#                             seq_rewards[:endpoint_pos+1] * seq_mask[:endpoint_pos+1]
#                         )
                        
#                         # 使用端点的value作为baseline
#                         endpoint_value = seq_values[endpoint_pos]
                        
#                         # 计算advantage (Algorithm 1, step 7)
#                         advantage = cumulative_reward - endpoint_value
#                         advantages[global_seq_idx, endpoint_pos] = advantage
#                         returns[global_seq_idx, endpoint_pos] = advantage + endpoint_value
    
#     return advantages, returns


# def _find_group_endpoints(mask, old_logprobs, new_logprobs, rewards, values):
#     """
#     Algorithm 2: 动态确定分组端点
#     """
#     endpoints = []
#     r_ppo = 0.0
#     r_grpo = 0.0
#     response_length = len(mask)
    
#     # 有效位置
#     valid_positions = torch.where(mask > 0)[0]
#     if len(valid_positions) == 0:
#         return endpoints
    
#     N = len(valid_positions)  # 序列长度
    
#     for step, t in enumerate(valid_positions):
#         t = t.item()
        
#         # 计算当前位置的advantage^2 (Â²_t)
#         current_advantage = rewards[t] - values[t]
#         advantage_squared = current_advantage ** 2
        
#         # 计算策略梯度范数的近似 ||∇π_θ(s_n)||²
#         policy_ratio = torch.exp(new_logprobs[t] - old_logprobs[t])
#         grad_norm_sq_approx = (policy_ratio - 1.0) ** 2
        
#         # Algorithm 2 公式计算
#         # r_ppo ← r_ppo + (1/N) ||∇π_θ(s_n)||² Â²_t
#         r_ppo += (1.0 / N) * grad_norm_sq_approx * advantage_squared
        
#         # r_grpo ← (1/N) ||∇(∏π_θ(s_n))||² Â²_t
#         if step == 0:
#             cumulative_policy_grad = grad_norm_sq_approx
#         else:
#             cumulative_policy_grad *= (1 + grad_norm_sq_approx)
        
#         r_grpo = (1.0 / N) * cumulative_policy_grad * advantage_squared
        
#         # Algorithm 2 判断条件: if n ≥ r_grpo/r_ppo then
#         n = step + 1  # 当前步数 (从1开始)
#         if r_ppo > 1e-8 and n >= (r_grpo / r_ppo):
#             endpoints.append(t)
#             # 重置累积器
#             r_ppo = 0.0
#             r_grpo = 0.0
    
#     # 确保最后一个有效位置是端点
#     last_valid_pos = valid_positions[-1].item()
#     if last_valid_pos not in endpoints:
#         endpoints.append(last_valid_pos)
    
#     return endpoints
@register_adv_est(AdvantageEstimator.GROUP)
def compute_group_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    values: torch.Tensor, 
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    config=None,
    **kwargs,
):
    """
    实现Group-based PPO算法 (增强监控版本)
    """
    batch_size, response_length = token_level_rewards.shape
    
    # 初始化
    advantages = torch.zeros_like(token_level_rewards)
    returns = torch.zeros_like(token_level_rewards)
    
    # 🆕 全局统计信息
    global_stats = {
        'total_sequences': batch_size,
        'total_groups': 0,
        'total_tokens': 0,
        'total_endpoints': 0,
        'group_details': [],
        'sequence_details': []
    }
    
    # 按组处理
    id2indices = defaultdict(list)
    for i in range(batch_size):
        id2indices[index[i]].append(i)
    
    global_stats['total_groups'] = len(id2indices)
    
    with torch.no_grad():
        for group_id, group_indices in id2indices.items():
            group_detail = {
                'group_id': str(group_id),
                'group_size': len(group_indices),
                'sequences': []
            }
            
            for seq_idx, global_seq_idx in enumerate(group_indices):
                seq_rewards = token_level_rewards[global_seq_idx]
                seq_values = values[global_seq_idx]
                seq_mask = response_mask[global_seq_idx]
                seq_old_logprob = old_log_prob[global_seq_idx]
                seq_new_logprob = log_prob[global_seq_idx]
                
                # 🆕 为每个序列收集debug信息
                debug_info = {}
                
                # Algorithm 2: 动态确定分组端点
                endpoints = _find_group_endpoints(
                    seq_mask, seq_old_logprob, seq_new_logprob, 
                    seq_rewards, seq_values, debug_info
                )
                
                # 统计信息
                valid_tokens = int(seq_mask.sum().item())
                global_stats['total_tokens'] += valid_tokens
                global_stats['total_endpoints'] += len(endpoints)
                
                # Algorithm 1: 只在端点计算advantage
                endpoint_advantages = []
                for endpoint_pos in endpoints:
                    if endpoint_pos < response_length and seq_mask[endpoint_pos] > 0:
                        # 计算到端点的累积奖励
                        cumulative_reward = torch.sum(
                            seq_rewards[:endpoint_pos+1] * seq_mask[:endpoint_pos+1]
                        )
                        
                        # 使用端点的value作为baseline
                        endpoint_value = seq_values[endpoint_pos]
                        
                        # 计算advantage (Algorithm 1, step 7)
                        advantage = cumulative_reward - endpoint_value
                        advantages[global_seq_idx, endpoint_pos] = advantage
                        returns[global_seq_idx, endpoint_pos] = advantage + endpoint_value
                        
                        endpoint_advantages.append({
                            'position': endpoint_pos,
                            'advantage': advantage.item(),
                            'cumulative_reward': cumulative_reward.item(),
                            'value': endpoint_value.item()
                        })
                
                # 🆕 记录序列详细信息
                seq_detail = {
                    'seq_idx': seq_idx,
                    'global_idx': global_seq_idx,
                    'valid_tokens': valid_tokens,
                    'num_endpoints': len(endpoints),
                    'endpoints': endpoints,
                    'endpoint_advantages': endpoint_advantages,
                    'compression_ratio': debug_info.get('compression_ratio', 0.0),
                    'step_details': debug_info.get('step_details', [])
                }
                
                group_detail['sequences'].append(seq_detail)
                global_stats['sequence_details'].append(seq_detail)
            
            global_stats['group_details'].append(group_detail)
    
    # 🆕 打印详细统计信息
    _print_group_statistics(global_stats)
    
    # 🆕 返回额外的监控指标
    monitoring_metrics = _compute_monitoring_metrics(global_stats, advantages, response_mask)
    
    return advantages, returns, monitoring_metrics
## v_2 增加监控
def _find_group_endpoints(mask, old_logprobs, new_logprobs, rewards, values, debug_info=None):
    """
    Algorithm 2: 动态确定分组端点 (增强debug版本)
    """
    endpoints = []
    r_ppo = 0.0
    r_grpo = 0.0
    response_length = len(mask)
    
    # 有效位置
    valid_positions = torch.where(mask > 0)[0]
    if len(valid_positions) == 0:
        return endpoints
    
    N = len(valid_positions)  # 序列长度
    
    # 🆕 Debug信息收集
    step_info = []
    
    for step, t in enumerate(valid_positions):
        t = t.item()
        
        # 计算当前位置的advantage^2 (Â²_t)
        current_advantage = rewards[t] - values[t]
        advantage_squared = current_advantage ** 2
        
        # 计算策略梯度范数的近似 ||∇π_θ(s_n)||²
        policy_ratio = torch.exp(new_logprobs[t] - old_logprobs[t])
        grad_norm_sq_approx = (policy_ratio - 1.0) ** 2
        
        # Algorithm 2 公式计算
        # r_ppo ← r_ppo + (1/N) ||∇π_θ(s_n)||² Â²_t
        r_ppo += (1.0 / N) * grad_norm_sq_approx * advantage_squared
        
        # r_grpo ← (1/N) ||∇(∏π_θ(s_n))||² Â²_t
        if step == 0:
            cumulative_policy_grad = grad_norm_sq_approx
        else:
            cumulative_policy_grad *= (1 + grad_norm_sq_approx)
        
        r_grpo = (1.0 / N) * cumulative_policy_grad * advantage_squared
        
        # 🆕 记录每步的详细信息
        step_detail = {
            'step': step + 1,
            'position': t,
            'advantage': current_advantage.item(),
            'policy_ratio': policy_ratio.item(),
            'r_ppo': r_ppo.item(),
            'r_grpo': r_grpo.item(),
            'ratio': (r_grpo / r_ppo).item() if r_ppo > 1e-8 else float('inf'),
            'is_endpoint': False
        }
        
        # Algorithm 2 判断条件: if n ≥ r_grpo/r_ppo then
        n = step + 1  # 当前步数 (从1开始)
        if r_ppo > 1e-8 and n >= (r_grpo / r_ppo):
            endpoints.append(t)
            step_detail['is_endpoint'] = True
            # 重置累积器
            r_ppo = 0.0
            r_grpo = 0.0
        
        step_info.append(step_detail)
    
    # 确保最后一个有效位置是端点
    last_valid_pos = valid_positions[-1].item()
    if last_valid_pos not in endpoints:
        endpoints.append(last_valid_pos)
        # 更新最后一步为端点
        if step_info:
            step_info[-1]['is_endpoint'] = True
    
    # 🆕 保存debug信息
    if debug_info is not None:
        debug_info.update({
            'total_length': N,
            'num_endpoints': len(endpoints),
            'endpoints': endpoints,
            'step_details': step_info,
            'compression_ratio': 1.0 - (len(endpoints) / N) if N > 0 else 0.0
        })
    
    return endpoints

def _print_group_statistics(stats):
    """打印GROUP算法的详细统计信息"""
    print("\n" + "="*80)
    print("🎯 GROUP算法运行统计")
    print("="*80)
    
    # 全局统计
    total_compression = 1.0 - (stats['total_endpoints'] / max(stats['total_tokens'], 1))
    print(f"📊 全局统计:")
    print(f"   - 总序列数: {stats['total_sequences']}")
    print(f"   - 总组数: {stats['total_groups']}")
    print(f"   - 总token数: {stats['total_tokens']}")
    print(f"   - 总端点数: {stats['total_endpoints']}")
    print(f"   - 压缩比: {total_compression:.2%}")
    print(f"   - 平均组大小: {stats['total_sequences'] / max(stats['total_groups'], 1):.1f}")
    print(f"   - 计算效率提升: {1.0 / (1.0 - total_compression + 1e-8):.1f}x")
    
    # 按组统计
    print(f"\n📈 分组详情:")
    for group_detail in stats['group_details'][:5]:  # 只显示前5组
        group_id = group_detail['group_id']
        group_size = group_detail['group_size']
        
        group_tokens = sum(seq['valid_tokens'] for seq in group_detail['sequences'])
        group_endpoints = sum(seq['num_endpoints'] for seq in group_detail['sequences'])
        group_compression = 1.0 - (group_endpoints / max(group_tokens, 1))
        
        print(f"   组 {group_id}: {group_size}个序列, {group_tokens}个token, {group_endpoints}个端点 (压缩率: {group_compression:.1%})")
        
        # 显示该组的序列详情
        for seq in group_detail['sequences'][:2]:  # 每组只显示前2个序列
            endpoints_str = ', '.join(map(str, seq['endpoints']))
            print(f"     序列{seq['seq_idx']}: {seq['valid_tokens']}token → {seq['num_endpoints']}端点 [{endpoints_str}]")
    
    if len(stats['group_details']) > 5:
        print(f"   ... (还有 {len(stats['group_details']) - 5} 个组)")
    
    # 端点分布统计
    endpoint_counts = [seq['num_endpoints'] for seq in stats['sequence_details']]
    if endpoint_counts:
        print(f"\n📊 端点分布:")
        print(f"   - 平均每序列端点数: {np.mean(endpoint_counts):.1f}")
        print(f"   - 端点数范围: {min(endpoint_counts)} ~ {max(endpoint_counts)}")
        print(f"   - 端点数标准差: {np.std(endpoint_counts):.1f}")
    
    print("="*80 + "\n")
    
def _compute_monitoring_metrics(global_stats, advantages, response_mask):
    """
    计算GROUP算法的监控指标
    
    Args:
        global_stats: 全局统计信息字典
        advantages: 计算得到的advantage张量 (batch_size, response_length)
        response_mask: response mask张量 (batch_size, response_length)
    
    Returns:
        dict: 包含各种监控指标的字典
    """
    metrics = {}
    
    # 基础统计指标
    total_tokens = global_stats['total_tokens']
    total_endpoints = global_stats['total_endpoints']
    total_sequences = global_stats['total_sequences']
    total_groups = global_stats['total_groups']
    
    # 避免除零
    safe_total_tokens = max(total_tokens, 1)
    safe_total_endpoints = max(total_endpoints, 1)
    safe_total_groups = max(total_groups, 1)
    
    # 🎯 核心效率指标
    compression_ratio = 1.0 - (total_endpoints / safe_total_tokens)
    metrics['group/compression_ratio'] = compression_ratio
    metrics['group/efficiency_gain'] = 1.0 / (1.0 - compression_ratio + 1e-8)
    metrics['group/memory_saved_ratio'] = compression_ratio
    
    # 📊 基础统计
    metrics['group/total_tokens'] = float(total_tokens)
    metrics['group/total_endpoints'] = float(total_endpoints)
    metrics['group/total_sequences'] = float(total_sequences)
    metrics['group/total_groups'] = float(total_groups)
    
    # 📈 分组统计
    metrics['group/avg_group_size'] = total_sequences / safe_total_groups
    metrics['group/avg_endpoints_per_sequence'] = total_endpoints / max(total_sequences, 1)
    metrics['group/avg_tokens_per_sequence'] = total_tokens / max(total_sequences, 1)
    
    # 🔍 端点分布分析
    endpoint_counts = [seq['num_endpoints'] for seq in global_stats['sequence_details']]
    if endpoint_counts:
        import numpy as np
        metrics['group/endpoint_count_mean'] = float(np.mean(endpoint_counts))
        metrics['group/endpoint_count_std'] = float(np.std(endpoint_counts))
        metrics['group/endpoint_count_min'] = float(np.min(endpoint_counts))
        metrics['group/endpoint_count_max'] = float(np.max(endpoint_counts))
        
        # 端点分布的分位数
        metrics['group/endpoint_count_p25'] = float(np.percentile(endpoint_counts, 25))
        metrics['group/endpoint_count_p50'] = float(np.percentile(endpoint_counts, 50))
        metrics['group/endpoint_count_p75'] = float(np.percentile(endpoint_counts, 75))
    
    # 🎨 压缩率分布分析
    compression_ratios = [seq.get('compression_ratio', 0.0) for seq in global_stats['sequence_details']]
    if compression_ratios:
        metrics['group/compression_ratio_mean'] = float(np.mean(compression_ratios))
        metrics['group/compression_ratio_std'] = float(np.std(compression_ratios))
        metrics['group/compression_ratio_min'] = float(np.min(compression_ratios))
        metrics['group/compression_ratio_max'] = float(np.max(compression_ratios))
    
    # 🧮 Advantage分析
    if advantages is not None and response_mask is not None:
        # 计算advantage的稀疏性
        valid_mask = response_mask.bool()
        nonzero_advantages = (advantages != 0.0) & valid_mask
        
        advantage_sparsity = 1.0 - (nonzero_advantages.sum().float() / valid_mask.sum().float()).item()
        metrics['group/advantage_sparsity'] = advantage_sparsity
        
        # 有效advantage的统计
        if nonzero_advantages.sum() > 0:
            valid_advantages = advantages[nonzero_advantages]
            metrics['group/advantage_mean'] = valid_advantages.mean().item()
            metrics['group/advantage_std'] = valid_advantages.std().item()
            metrics['group/advantage_abs_mean'] = valid_advantages.abs().mean().item()
            
            # Advantage的分布
            metrics['group/advantage_positive_ratio'] = (valid_advantages > 0).float().mean().item()
            metrics['group/advantage_negative_ratio'] = (valid_advantages < 0).float().mean().item()
        else:
            metrics['group/advantage_mean'] = 0.0
            metrics['group/advantage_std'] = 0.0
            metrics['group/advantage_abs_mean'] = 0.0
            metrics['group/advantage_positive_ratio'] = 0.0
            metrics['group/advantage_negative_ratio'] = 0.0
    
    # 🔬 分组质量分析
    group_sizes = [group['group_size'] for group in global_stats['group_details']]
    if group_sizes:
        metrics['group/group_size_mean'] = float(np.mean(group_sizes))
        metrics['group/group_size_std'] = float(np.std(group_sizes))
        metrics['group/group_size_min'] = float(np.min(group_sizes))
        metrics['group/group_size_max'] = float(np.max(group_sizes))
        
        # 分组平衡性 (标准差越小越平衡)
        metrics['group/group_balance_score'] = 1.0 - (np.std(group_sizes) / (np.mean(group_sizes) + 1e-8))
    
    # ⚡ 性能预估指标
    if total_tokens > 0 and total_endpoints > 0:
        # 理论计算节省
        compute_savings = (total_tokens - total_endpoints) / total_tokens
        metrics['group/compute_savings'] = compute_savings
        
        # 内存节省预估
        memory_savings = compression_ratio * 0.8  # 考虑overhead
        metrics['group/memory_savings_est'] = memory_savings
        
        # 训练加速预估
        speedup_factor = 1.0 / (1.0 - compute_savings * 0.7)  # 考虑其他overhead
        metrics['group/speedup_factor_est'] = speedup_factor
    
    # 🏷️ 算法健康度指标
    # 检查是否有异常的分组情况
    if endpoint_counts:
        # 如果大多数序列都只有1个端点，可能算法退化了
        single_endpoint_ratio = sum(1 for count in endpoint_counts if count == 1) / len(endpoint_counts)
        metrics['group/single_endpoint_ratio'] = single_endpoint_ratio
        
        # 如果压缩率太低，算法效果不好
        metrics['group/algorithm_effectiveness'] = max(0.0, compression_ratio - 0.1) / 0.9
        
        # 分组一致性 (组内端点数的一致性)
        if len(global_stats['group_details']) > 0:
            group_consistency_scores = []
            for group in global_stats['group_details']:
                seq_endpoint_counts = [seq['num_endpoints'] for seq in group['sequences']]
                if len(seq_endpoint_counts) > 1:
                    consistency = 1.0 - (np.std(seq_endpoint_counts) / (np.mean(seq_endpoint_counts) + 1e-8))
                    group_consistency_scores.append(max(0.0, consistency))
            
            if group_consistency_scores:
                metrics['group/grouping_consistency'] = float(np.mean(group_consistency_scores))
            else:
                metrics['group/grouping_consistency'] = 1.0
    
    # 🔍 Debug级别的详细指标 (可选)
    debug_mode = len(global_stats.get('sequence_details', [])) > 0
    if debug_mode:
        # 每个组的详细统计
        for i, group in enumerate(global_stats['group_details'][:5]):  # 只记录前5个组
            group_tokens = sum(seq['valid_tokens'] for seq in group['sequences'])
            group_endpoints = sum(seq['num_endpoints'] for seq in group['sequences'])
            group_compression = 1.0 - (group_endpoints / max(group_tokens, 1))
            
            metrics[f'group/group_{i}_size'] = float(group['group_size'])
            metrics[f'group/group_{i}_compression'] = group_compression
            metrics[f'group/group_{i}_tokens'] = float(group_tokens)
            metrics[f'group/group_{i}_endpoints'] = float(group_endpoints)
    
    # 🎯 核心KPI汇总 (用于重点监控)
    metrics['group/kpi_compression'] = compression_ratio
    metrics['group/kpi_efficiency'] = metrics['group/efficiency_gain']
    metrics['group/kpi_health'] = metrics.get('group/algorithm_effectiveness', 0.5)
    
    return metrics