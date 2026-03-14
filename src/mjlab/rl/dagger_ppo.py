"""DAgger-style PPO: PPO + KL(teacher || student) distillation loss.

Implementation follows TWIST on_policy_dagger_runner / DaggerPPO:
- Student acts in env on policy obs; value uses critic obs.
- Teacher runs on critic (privileged) obs; KL loss encourages student to match teacher.
- dagger_coef is annealed (cosine) to dagger_coef_min over dagger_coef_anneal_steps.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO


def _kl_gaussian(
  mu_s: torch.Tensor,
  sigma_s: torch.Tensor,
  mu_t: torch.Tensor,
  sigma_t: torch.Tensor,
) -> torch.Tensor:
  """KL(student || teacher) per action dimension; sum over actions."""
  return (
    torch.log(sigma_t / sigma_s + 1e-8)
    + (sigma_s**2 + (mu_s - mu_t) ** 2) / (2 * sigma_t**2 + 1e-8)
    - 0.5
  )


def cosine_decay_weight(init_weight: float, step: int, total_steps: int) -> float:
  return init_weight * (0.5 * (1 + math.cos(math.pi * step / total_steps)))


class DaggerPPO(PPO):
  """PPO + KL(teacher || student)（双 Teacher 蒸馏）.

  前摔/后摔各一个 Teacher；fall_direction（critic obs 最后一维）按样本选择用哪个 Teacher。
  """

  def __init__(
    self,
    *,
    teacher_forward_actor: nn.Module,
    teacher_backward_actor: nn.Module,
    dagger_coef: float = 0.1,
    dagger_coef_anneal_steps: int = 30_000,
    dagger_coef_min: float = 0.01,
    eval_student: bool = False,
    **kwargs,
  ) -> None:
    for k in (
      "teacher_forward_actor",
      "teacher_backward_actor",
      "dagger_coef",
      "dagger_coef_anneal_steps",
      "dagger_coef_min",
      "eval_student",
    ):
      kwargs.pop(k, None)
    super().__init__(**kwargs)
    # PPO has .policy (ActorCritic); DaggerPPO uses .actor / .critic in update()
    self.actor = getattr(self, "actor", None) or getattr(self.policy, "actor", None)
    self.critic = getattr(self, "critic", None) or getattr(self.policy, "critic", None)
    self.teacher_forward_actor = teacher_forward_actor.to(self.device)
    self.teacher_backward_actor = teacher_backward_actor.to(self.device)
    self._eval_student = eval_student
    self._dagger_coef_init = dagger_coef
    self._dagger_coef = dagger_coef
    self._dagger_coef_anneal_steps = dagger_coef_anneal_steps
    self._dagger_coef_min = dagger_coef_min
    self._dagger_update_counter = 0

  def update(self) -> dict[str, float]:
    """Run PPO update with added KL(teacher || student) loss; anneal dagger_coef."""
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_kl_teacher_student = 0.0
    mean_rnd_loss = 0.0 if self.rnd else None  # type: ignore[attr-defined]
    mean_symmetry_loss = 0.0 if self.symmetry else None  # type: ignore[attr-defined]

    if getattr(self.policy, "is_recurrent", False):
      generator = self.storage.recurrent_mini_batch_generator(  # type: ignore[attr-defined]
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(  # type: ignore[attr-defined]
        self.num_mini_batches, self.num_learning_epochs
      )

    for (
      obs_batch,
      actions_batch,
      target_values_batch,
      advantages_batch,
      returns_batch,
      old_actions_log_prob_batch,
      old_mu_batch,
      old_sigma_batch,
      hid_states_batch,
      masks_batch,
    ) in generator:
      # rsl_rl mini_batch_generator yields a 10-tuple; obs may have .batch_size or .shape
      original_batch_size = (
        obs_batch.batch_size[0]
        if getattr(obs_batch, "batch_size", None) is not None
        else obs_batch.shape[0]
      )

      if self.normalize_advantage_per_mini_batch:  # type: ignore[attr-defined]
        with torch.no_grad():
          advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
          )

      if self.symmetry and self.symmetry["use_data_augmentation"]:  # type: ignore[attr-defined]
        data_augmentation_func = self.symmetry["data_augmentation_func"]  # type: ignore[attr-defined]
        obs_batch, actions_batch = data_augmentation_func(
          env=self.symmetry["_env"],  # type: ignore[attr-defined]
          obs=obs_batch,
          actions=actions_batch,
        )
        num_aug = int(
          (obs_batch.batch_size[0] if getattr(obs_batch, "batch_size", None) else obs_batch.shape[0])
          / original_batch_size
        )
        old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
        target_values_batch = target_values_batch.repeat(num_aug, 1)
        advantages_batch = advantages_batch.repeat(num_aug, 1)
        returns_batch = returns_batch.repeat(num_aug, 1)

      self.policy.act(
        obs_batch,
        masks=masks_batch,
        hidden_states=hid_states_batch[0] if hid_states_batch else None,
        stochastic_output=True,
      )
      actions_log_prob = self.policy.get_actions_log_prob(actions_batch)
      values = self.policy.evaluate(
        obs_batch,
        masks=masks_batch,
        hidden_states=hid_states_batch[1] if hid_states_batch else None,
      )
      distribution_params = (
        self.policy.action_mean[:original_batch_size],
        self.policy.action_std[:original_batch_size],
      )
      entropy = self.policy.entropy[:original_batch_size]

      if (
        self.desired_kl is not None  # type: ignore[attr-defined]
        and self.schedule == "adaptive"  # type: ignore[attr-defined]
      ):
        with torch.inference_mode():
          kl = torch.sum(
            torch.log(old_sigma_batch / self.policy.action_std[:original_batch_size] + 1e-5)
            + (
              torch.square(self.policy.action_std[:original_batch_size])
              + torch.square(self.policy.action_mean[:original_batch_size] - old_mu_batch)
            )
            / (2.0 * torch.square(old_sigma_batch))
            - 0.5,
            dim=-1,
          )
          kl_mean = torch.mean(kl)
        if self.is_multi_gpu:  # type: ignore[attr-defined]
          torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
          kl_mean /= self.gpu_world_size  # type: ignore[attr-defined]
        if self.gpu_global_rank == 0:  # type: ignore[attr-defined]
          # 蒸馏时 KL(teacher||student) 会使 policy 变化大，LR 易被压到下限；下限用 1e-4 避免完全学不动
          lr_floor = 1e-4
          if kl_mean > self.desired_kl * 2.0:  # type: ignore[attr-defined]
            self.learning_rate = max(lr_floor, self.learning_rate / 1.5)
          elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:  # type: ignore[attr-defined]
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        if self.is_multi_gpu:  # type: ignore[attr-defined]
          lr_tensor = torch.tensor(self.learning_rate, device=self.device)
          torch.distributed.broadcast(lr_tensor, src=0)
          self.learning_rate = lr_tensor.item()
        for param_group in self.optimizer.param_groups:
          param_group["lr"] = self.learning_rate

      ratio = torch.exp(
        actions_log_prob - torch.squeeze(old_actions_log_prob_batch)
      )
      surrogate = -torch.squeeze(advantages_batch) * ratio
      surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:  # type: ignore[attr-defined]
        value_clipped = target_values_batch + (values - target_values_batch).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (returns_batch - values).pow(2).mean()

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss  # type: ignore[attr-defined]
        - self.entropy_coef * entropy.mean()  # type: ignore[attr-defined]
      )

      kl_ts = self._compute_kl_teacher_student(
        obs_batch, distribution_params, original_batch_size
      )
      if kl_ts is not None:
        loss = loss + self._dagger_coef * kl_ts.mean()
        mean_kl_teacher_student += kl_ts.mean().item()

      if self.symmetry:  # type: ignore[attr-defined]
        data_augmentation_func = self.symmetry["data_augmentation_func"]  # type: ignore[attr-defined]
        if not self.symmetry["use_data_augmentation"]:  # type: ignore[attr-defined]
          obs_batch, _ = data_augmentation_func(
            obs=obs_batch, actions=None, env=self.symmetry["_env"]  # type: ignore[attr-defined]
          )
        self.policy.act(obs_batch)
        mean_actions = self.policy.action_mean
        action_mean_orig = mean_actions[:original_batch_size]
        _, actions_mean_symm = data_augmentation_func(
          obs=None, actions=action_mean_orig, env=self.symmetry["_env"]  # type: ignore[attr-defined]
        )
        mse_loss = nn.MSELoss()
        symmetry_loss = mse_loss(
          mean_actions[original_batch_size:],
          actions_mean_symm.detach()[original_batch_size:],
        )
        if self.symmetry["use_mirror_loss"]:  # type: ignore[attr-defined]
          loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss  # type: ignore[attr-defined]
        else:
          symmetry_loss = symmetry_loss.detach()
        if mean_symmetry_loss is not None:
          mean_symmetry_loss += symmetry_loss.item()

      rnd_loss = None
      if self.rnd:  # type: ignore[attr-defined]
        with torch.no_grad():
          obs_slice = obs_batch[:original_batch_size] if hasattr(obs_batch, "__getitem__") else obs_batch
          rnd_state = self.rnd.get_rnd_state(obs_slice)  # type: ignore[attr-defined]
          rnd_state = self.rnd.state_normalizer(rnd_state)  # type: ignore[attr-defined]
        predicted_embedding = self.rnd.predictor(rnd_state)  # type: ignore[attr-defined]
        target_embedding = self.rnd.target(rnd_state).detach()  # type: ignore[attr-defined]
        rnd_loss = nn.functional.mse_loss(predicted_embedding, target_embedding)
        if mean_rnd_loss is not None:
          mean_rnd_loss += rnd_loss.item()

      self.optimizer.zero_grad()
      loss.backward()
      if self.rnd and rnd_loss is not None:  # type: ignore[attr-defined]
        self.rnd_optimizer.zero_grad()  # type: ignore[attr-defined]
        rnd_loss.backward()
      if self.is_multi_gpu:  # type: ignore[attr-defined]
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)  # type: ignore[attr-defined]
      self.optimizer.step()
      if self.rnd and self.rnd_optimizer:  # type: ignore[attr-defined]
        self.rnd_optimizer.step()  # type: ignore[attr-defined]

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    mean_kl_teacher_student /= num_updates
    if mean_rnd_loss is not None:
      mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
      mean_symmetry_loss /= num_updates

    self.storage.clear()  # type: ignore[attr-defined]
    self._dagger_update_counter += 1
    if self._dagger_update_counter < self._dagger_coef_anneal_steps:
      # 从 _dagger_coef_init 退火到 _dagger_coef_min（cosine 1→0）
      alpha = cosine_decay_weight(1.0, self._dagger_update_counter, self._dagger_coef_anneal_steps)
      self._dagger_coef = self._dagger_coef_min + (self._dagger_coef_init - self._dagger_coef_min) * alpha
    else:
      self._dagger_coef = self._dagger_coef_min

    loss_dict: dict[str, float] = {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
      "kl_teacher_student": mean_kl_teacher_student,
    }
    if self.rnd:  # type: ignore[attr-defined]
      loss_dict["rnd"] = mean_rnd_loss  # type: ignore[assignment]
    if self.symmetry:  # type: ignore[attr-defined]
      loss_dict["symmetry"] = mean_symmetry_loss  # type: ignore[assignment]
    return loss_dict

  def _compute_kl_teacher_student(
    self,
    obs,
    distribution_params: tuple,
    original_batch_size: int,
  ) -> torch.Tensor | None:
    """Compute KL(teacher || student); use fall_direction to select 前摔/后摔 teacher."""
    if self._eval_student:
      return None
    try:
      obs_size = obs.batch_size[0] if getattr(obs, "batch_size", None) is not None else obs.shape[0]
      if original_batch_size < obs_size:
        obs = obs[:original_batch_size]
      return self._compute_kl_dual_teacher(
        obs, distribution_params, original_batch_size
      )
    except Exception:
      return None

  def _compute_kl_dual_teacher(
    self,
    obs,
    distribution_params: tuple,
    original_batch_size: int,
  ) -> torch.Tensor | None:
    """Use fall_direction (last dim of critic obs) to select 前摔/后摔 teacher per sample.
    obs 须含 "critic"（取 fall_direction）与 "teacher"（Teacher 前向输入）；rollout 存的是 env.get_observations() 的完整 dict。"""
    try:
      if not hasattr(obs, "get") or "critic" not in obs.keys():
        return None
      if "teacher" not in obs.keys():
        return None
      critic_obs = obs["critic"]
      if original_batch_size < critic_obs.shape[0]:
        critic_obs = critic_obs[:original_batch_size]
      fall_dir = critic_obs[:, -1]  # 依赖 env 中 critic 观测最后一维为 fall_direction
      self.teacher_forward_actor.eval()
      self.teacher_backward_actor.eval()
      with torch.no_grad():
        _ = self.teacher_forward_actor.act(obs)
        mu_f = self.teacher_forward_actor.action_mean.detach()
        sigma_f = self.teacher_forward_actor.action_std.detach().clamp(min=1e-6)
        _ = self.teacher_backward_actor.act(obs)
        mu_b = self.teacher_backward_actor.action_mean.detach()
        sigma_b = self.teacher_backward_actor.action_std.detach().clamp(min=1e-6)
      mask = (fall_dir >= 0).float().unsqueeze(-1)
      mu_t = mask * mu_f + (1 - mask) * mu_b
      sigma_t = mask * sigma_f + (1 - mask) * sigma_b
      mu_s = distribution_params[0]
      sigma_s = distribution_params[1].clamp(min=1e-6)
      kl_per_dim = _kl_gaussian(mu_s, sigma_s, mu_t, sigma_t)
      return kl_per_dim.sum(dim=-1)
    except Exception:
      return None
