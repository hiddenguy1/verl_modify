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
    
        # import pdb
        # pdb.set_trace()
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
    """
    # ## 新增: 端点mask处理
    # 如果group_endpoints不为None，则只在端点位置计算损失
    if group_endpoints is not None:
        # group_endpoints: (batch, seq_len) 0/1 mask, 1表示端点
        effective_mask = response_mask * group_endpoints
    else:
    # import pdb
    # pdb.set_trace()
        effective_mask = response_mask
    # ## end新增

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    # ## 新增: 用effective_mask替换response_mask
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, effective_mask)
    # ## end新增
    # ppo_kl = verl_F.masked_mean(-(log_prob - old_log_prob), response_mask)
    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    # ## 新增: 用effective_mask替换response_mask
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), effective_mask)
    # ## end新增

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), effective_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    # ## 新增: 用effective_mask替换response_mask
    # import pdb
    # pdb.set_trace()
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=effective_mask, loss_agg_mode=loss_agg_mode)
    # ## end新增

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
    # import pdb
    # pdb.set_trace()
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
# Group_PPO说明：
# 1. advantage一律用GAE（compute_gae_advantage_return）计算，所有token都要有。
# 2. 只在分组端点（endpoints）上做PPO损失和反向传播，其余token直接mask掉或不参与loss计算。
# 3. _find_group_endpoints用于生成端点mask。
## end新增

## 新增
@register_adv_est(AdvantageEstimator.GROUP)
def  compute_group_advantage(
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
    新增功能：分组内advantage广播
    """
    batch_size, response_length = token_level_rewards.shape
    
    # 初始化
    advantages = torch.zeros_like(token_level_rewards)
    returns = torch.zeros_like(token_level_rewards)
    
    # 🆕 新增：分组mask，用于标识每个token属于哪个分组
    group_mask = torch.zeros_like(token_level_rewards, dtype=torch.long)
    
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
    
    # 🆕 创建group_id到整数的映射，确保所有group_id都是整数
    unique_group_ids = list(id2indices.keys())
    group_id_to_int = {}
    for idx, group_id in enumerate(unique_group_ids):
        if isinstance(group_id, str):
            # 如果是字符串，尝试转换为整数，如果失败则使用索引
            try:
                group_id_to_int[group_id] = int(group_id)
            except ValueError:
                group_id_to_int[group_id] = idx + 1
        else:
            group_id_to_int[group_id] = int(group_id)
    
    with torch.no_grad():
        for group_id, group_indices in id2indices.items():
            group_detail = {
                'group_id': str(group_id),
                'group_size': len(group_indices),
                'sequences': []
            }
            group_endpoint_advantages = []
            group_endpoint_positions = []
            for seq_idx, global_seq_idx in enumerate(group_indices):
                seq_rewards = token_level_rewards[global_seq_idx]
                seq_values = values[global_seq_idx]
                seq_mask = response_mask[global_seq_idx]
                seq_old_logprob = old_log_prob[global_seq_idx]
                seq_new_logprob = log_prob[global_seq_idx]
                debug_info = {}
                endpoints = _find_group_endpoints(
                    seq_mask, seq_old_logprob, seq_new_logprob, 
                    seq_rewards, seq_values, debug_info
                )
                valid_tokens = int(seq_mask.sum().item())
                global_stats['total_tokens'] += valid_tokens
                global_stats['total_endpoints'] += len(endpoints)
                # === 修正分组mask为token级分组ID ===
                # endpoints为分组末端，分组区间为[start:end]
                last_pos = -1
                for seg_idx, endpoint_pos in enumerate(endpoints):
                    if endpoint_pos < response_length and seq_mask[endpoint_pos] > 0:
                        # 分组区间：[last_pos+1, endpoint_pos]
                        start_pos = last_pos + 1
                        end_pos = endpoint_pos + 1
                        
                        group_mask[global_seq_idx, start_pos:end_pos] = seg_idx + 1
                        
                        # 修正：只计算当前分组的累积奖励
                        cumulative_reward = torch.sum(
                            seq_rewards[start_pos:end_pos] * seq_mask[start_pos:end_pos]
                        )
                        endpoint_value = seq_values[endpoint_pos]
                        advantage = cumulative_reward - endpoint_value
                        # import pdb
                        # pdb.set_trace()
                        group_endpoint_advantages.append(advantage)
                        group_endpoint_positions.append((global_seq_idx, endpoint_pos))
                        returns[global_seq_idx, endpoint_pos] = advantage + endpoint_value
                    last_pos = endpoint_pos
                # === END 修正 ===
                endpoint_advantages = []
                for endpoint_pos in endpoints:
                    if endpoint_pos < response_length and seq_mask[endpoint_pos] > 0:
                        cumulative_reward = torch.sum(
                            seq_rewards[:endpoint_pos+1] * seq_mask[:endpoint_pos+1]
                        )
                        endpoint_value = seq_values[endpoint_pos]
                        advantage = cumulative_reward - endpoint_value
                        endpoint_advantages.append({
                            'position': endpoint_pos,
                            'advantage': advantage.item(),
                            'cumulative_reward': cumulative_reward.item(),
                            'value': endpoint_value.item()
                        })
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
            if group_endpoint_advantages:
                # 修正后的advantage广播逻辑
                for global_seq_idx in group_indices:
                    seq_mask = response_mask[global_seq_idx]
                    seq_group_mask = group_mask[global_seq_idx]
                    
                    # 初始化该序列的advantage
                    seq_advantages = torch.zeros_like(seq_mask, dtype=torch.float)
                    
                    # 为每个分组分配对应的advantage
                    for seg_idx, endpoint_advantage in enumerate(group_endpoint_advantages):
                        group_id = seg_idx + 1
                        # 找到属于该分组的token位置
                        group_tokens = (seq_group_mask == group_id) & (seq_mask > 0)
                        if group_tokens.any():
                            seq_advantages[group_tokens] = endpoint_advantage
                    
                    advantages[global_seq_idx] = seq_advantages
            global_stats['group_details'].append(group_detail)

    # 🆕 打印详细统计信息
    _print_group_statistics(global_stats)
    
    # 🆕 返回额外的监控指标
    monitoring_metrics = _compute_monitoring_metrics(global_stats, advantages, response_mask)
    
    # 🆕 新增：返回分组mask，用于后续训练
    
    return advantages, returns, monitoring_metrics, group_mask
## v_2 增加监控
def _find_group_endpoints(mask, old_logprobs, new_logprobs, rewards, values, debug_info=None):
    """
    Algorithm 2: 动态确定分组端点 (完全按照论文实现)
    
    论文中的Algorithm 2 Group:
    - r_ppo ← (1/N) || Σ_{n=1}^t Â_n ⋅ ∇ ln π_θ(s_n) ||^2
    - r_grpo ← (1/N) || Σ_{n=1}^t ∇ ln π_θ(s_n) ||^2 Â_t^2
    - if t ≥ r_grpo / r_ppo then: 重置累积器
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
    
    # 累积梯度向量
    cumulative_advantage_grad = torch.zeros_like(old_logprobs)
    cumulative_grad = torch.zeros_like(old_logprobs)
    
    for step, t in enumerate(valid_positions):
        t = t.item()
        
        # 计算当前位置的advantage (Â_t)
        current_advantage = rewards[t] - values[t]
        
        # 计算策略梯度 ∇ ln π_θ(s_t)
        policy_grad = new_logprobs[t] - old_logprobs[t]  # ∇ ln π_θ(s_t)
        
        # 累积梯度向量 (从1到t)
        t_idx = int(t)
        if t_idx > 0:
            cumulative_advantage_grad[t_idx] = cumulative_advantage_grad[t_idx-1] + current_advantage * policy_grad
            cumulative_grad[t_idx] = cumulative_grad[t_idx-1] + policy_grad
        else:
            cumulative_advantage_grad[t_idx] = current_advantage * policy_grad
            cumulative_grad[t_idx] = policy_grad
        
        # 论文公式计算
        # r_ppo ← (1/N) || Σ_{n=1}^t Â_n ⋅ ∇ ln π_θ(s_n) ||^2
        r_ppo = (1.0 / N) * (cumulative_advantage_grad[t_idx] ** 2)
        
        # r_grpo ← (1/N) || Σ_{n=1}^t ∇ ln π_θ(s_n) ||^2 Â_t^2
        r_grpo = (1.0 / N) * (cumulative_grad[t_idx] ** 2) * (current_advantage ** 2)
        
        # 🆕 记录每步的详细信息
        step_detail = {
            'step': step + 1,
            'position': t,
            'advantage': current_advantage.item(),
            'policy_grad': policy_grad.item(),
            'r_ppo': r_ppo.item(),
            'r_grpo': r_grpo.item(),
            'ratio': (r_grpo / r_ppo).item() if r_ppo > 1e-8 else float('inf'),
            'is_endpoint': False
        }
        
        # Algorithm 2 判断条件: if t ≥ r_grpo/r_ppo then
        if r_ppo > 1e-8 and (step + 1) >= (r_grpo / r_ppo):
            endpoints.append(t)
            step_detail['is_endpoint'] = True
            # 重置累积器
            r_ppo = 0.0
            r_grpo = 0.0
            # 重置累积梯度向量
            cumulative_advantage_grad = torch.zeros_like(old_logprobs)
            cumulative_grad = torch.zeros_like(old_logprobs)
        
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


def create_critic_endpoint_mask(response_mask: torch.Tensor, group_mask: torch.Tensor) -> torch.Tensor:
    """
    为Critic训练创建端点mask
    
    Args:
        response_mask: (batch_size, response_length) 响应mask
        group_mask: (batch_size, response_length) 分组mask
        
    Returns:
        endpoint_mask: (batch_size, response_length) 端点mask，只在每个分组的最后一个有效token位置为True
    """
    endpoint_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    batch_size = response_mask.shape[0]
    
    for i in range(batch_size):
        seq_mask = response_mask[i]
        seq_group_mask = group_mask[i]
        
        # 找到该序列中每个分组的最后一个有效token
        valid_positions = torch.where(seq_mask > 0)[0]
        if len(valid_positions) > 0:
            # 按分组找到每个分组的最后一个位置
            unique_groups = torch.unique(seq_group_mask[valid_positions])
            for group_id in unique_groups:
                if group_id == 0:  # 跳过无效分组
                    continue
                # 找到该分组在该序列中的最后一个位置
                group_positions = valid_positions[seq_group_mask[valid_positions] == group_id]
                if len(group_positions) > 0:
                    last_group_pos = group_positions[-1]
                    endpoint_mask[i, last_group_pos] = True
    
    return endpoint_mask
