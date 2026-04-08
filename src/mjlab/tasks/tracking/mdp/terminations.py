from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import MotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )


def bad_anchor_ori(
  env: ManagerBasedRlEnv, asset_cfg: "SceneEntityCfg", command_name: str, threshold: float
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )

  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )

  return (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  return torch.any(error > threshold, dim=-1)


def bad_torso_z_vs_motion_ref(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: "SceneEntityCfg",
  body_name: str,
  threshold: float,
) -> torch.Tensor:
  """|z_motion - z_robot| in world frame for a single body (e.g. torso)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset: Entity = env.scene[asset_cfg.name]
  try:
    cmd_idx = command.cfg.body_names.index(body_name)
  except ValueError as e:
    raise ValueError(
      f"bad_torso_z_vs_motion_ref: {body_name!r} not in motion body_names"
    ) from e
  try:
    rob_idx = asset.body_names.index(body_name)
  except ValueError as e:
    raise ValueError(
      f"bad_torso_z_vs_motion_ref: {body_name!r} not in asset body_names"
    ) from e
  z_ref = command.body_pos_w[:, cmd_idx, 2]
  z_rob = asset.data.body_link_pos_w[:, rob_idx, 2]
  return (z_ref - z_rob).abs() > float(threshold)


def _pre_recovery_only_terminate(
  env: ManagerBasedRlEnv, condition: torch.Tensor
) -> torch.Tensor:
  """Recovery 课程未开启：条件成立则终止回合。开启 recovery 后：本条不再判终/进 recovery。"""
  env_any = cast(Any, env)
  _ensure_recovery_state(env)
  if bool(env_any.recovery_enabled):
    return torch.zeros_like(condition)
  return condition


def _pre_recovery_terminate_or_post_recovery_enter(
  env: ManagerBasedRlEnv, condition: torch.Tensor
) -> torch.Tensor:
  """Before recovery curriculum: terminate; after enabled: enter recovery."""
  env_any = cast(Any, env)
  _ensure_recovery_state(env)
  if bool(env_any.recovery_enabled):
    return _maybe_start_recovery_from_condition(env, condition)
  return condition


def _recovery_phase_enter_only(
  env: ManagerBasedRlEnv, condition: torch.Tensor
) -> torch.Tensor:
  """Recovery 课程未开启：不生效。开启后：条件成立则进入 recovery（或已在 recovery 则抑制）。"""
  env_any = cast(Any, env)
  _ensure_recovery_state(env)
  if not bool(env_any.recovery_enabled):
    return torch.zeros_like(condition)
  return _maybe_start_recovery_from_condition(env, condition)


def _ensure_recovery_state(env: ManagerBasedRlEnv) -> None:
  """Lazily initialize per-env recovery buffers.

  We keep this state inside `env` so multiple termination terms and command
  playback can coordinate without changing the base env implementation.
  """
  env_any = cast(Any, env)
  if not hasattr(env_any, "recovery_enabled"):
    env_any.recovery_enabled = False

  # Buffers are created lazily (env may be constructed before termination
  # functions are ever called).
  if not hasattr(env_any, "recovery_mode_buf"):
    env_any.recovery_mode_buf = torch.zeros(
      env_any.num_envs, device=env_any.device, dtype=torch.bool
    )
  if not hasattr(env_any, "recovery_start_step_buf"):
    env_any.recovery_start_step_buf = torch.zeros(
      env_any.num_envs, device=env_any.device, dtype=torch.long
    )
  if not hasattr(env_any, "recovery_stable_count_buf"):
    env_any.recovery_stable_count_buf = torch.zeros(
      env_any.num_envs, device=env_any.device, dtype=torch.long
    )
  if not hasattr(env_any, "recovery_success_bonus_buf"):
    env_any.recovery_success_bonus_buf = torch.zeros(
      env_any.num_envs, device=env_any.device, dtype=torch.float32
    )
  if not hasattr(env_any, "recovery_entry_penalty_buf"):
    env_any.recovery_entry_penalty_buf = torch.zeros(
      env_any.num_envs, device=env_any.device, dtype=torch.float32
    )
  if not hasattr(env_any, "recovery_recovered_once_buf"):
    env_any.recovery_recovered_once_buf = torch.zeros(
      env_any.num_envs, device=env_any.device, dtype=torch.bool
    )
  if not hasattr(env_any, "recovery_enter_total_count"):
    env_any.recovery_enter_total_count = 0
  if not hasattr(env_any, "recovery_success_total_count"):
    env_any.recovery_success_total_count = 0

  # Clear one-step success bonus once per env step.
  last_bonus_clear = getattr(env_any, "_recovery_last_bonus_clear_common_step_counter", None)
  if last_bonus_clear != env_any.common_step_counter:
    env_any.recovery_success_bonus_buf.zero_()
    env_any._recovery_last_bonus_clear_common_step_counter = env_any.common_step_counter

  # Clear recovery state on reset.
  # In `env.step()`, `episode_length_buf` is incremented before termination
  # functions are evaluated, so just-after-reset values show up as `1` here.
  reset_mask = env_any.episode_length_buf <= 1
  if reset_mask.any():
    # TerminationManager may call multiple termination terms in a single
    # step; clear only once per env step to avoid wiping recovery that was
    # just started by an earlier term in the same step.
    last_cleared = getattr(
      env_any, "_recovery_last_clear_common_step_counter", None
    )
    if last_cleared != env_any.common_step_counter:
      env_any.recovery_mode_buf[reset_mask] = False
      env_any.recovery_start_step_buf[reset_mask] = 0
      env_any.recovery_stable_count_buf[reset_mask] = 0
      env_any.recovery_success_bonus_buf[reset_mask] = 0.0
      env_any.recovery_entry_penalty_buf[reset_mask] = 0.0
      env_any.recovery_recovered_once_buf[reset_mask] = False
      env_any._recovery_last_clear_common_step_counter = env_any.common_step_counter

  # Always expose the mask to the AMP runner (it will gate reward mixing).
  env_any.extras["recovery_mask"] = env_any.recovery_mode_buf.clone()
  env_any.extras["recovery_success_bonus"] = env_any.recovery_success_bonus_buf.clone()


def _maybe_start_recovery_from_condition(
  env: ManagerBasedRlEnv,
  condition: torch.Tensor,
) -> torch.Tensor:
  """在 ``recovery_enabled`` 下：满足 condition 则进入 recovery，本步不终局。

  调用方应保证仅在 recovery 课程已开启时使用（例如
  :func:`_recovery_phase_enter_only`）。新进入的 env 会写入 ``recovery_entry_penalty_buf=1``。
  """
  env_any = cast(Any, env)
  _ensure_recovery_state(env)

  assert bool(env_any.recovery_enabled), (
    "_maybe_start_recovery_from_condition expects env.recovery_enabled True"
  )

  recovery_mode = env_any.recovery_mode_buf
  # Start recovery only for envs that are not already recovering.
  enter_mask = condition & ~recovery_mode
  enter_count = int(torch.count_nonzero(enter_mask).item())
  reenter_count = int(
    torch.count_nonzero(enter_mask & env_any.recovery_recovered_once_buf).item()
  )
  if enter_mask.any():
    env_any.recovery_mode_buf[enter_mask] = True
    env_any.recovery_start_step_buf[enter_mask] = env_any.episode_length_buf[
      enter_mask
    ]
    env_any.recovery_stable_count_buf[enter_mask] = 0
    env_any.recovery_entry_penalty_buf[enter_mask] = 1.0
  env_any.recovery_enter_total_count += enter_count

  log = env_any.extras.setdefault("log", {})
  log["Metrics/recovery_enter_count_step"] = float(enter_count)
  log["Metrics/recovery_reenter_after_success_count_step"] = float(reenter_count)
  denom = max(float(env_any.recovery_enter_total_count), 1.0)
  log["Metrics/recovery_success_rate_total"] = (
    float(env_any.recovery_success_total_count) / denom
  )

  # During recovery mode, we suppress termination from the original terms.
  return torch.zeros_like(condition)


def recovery_or_terminate_bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  """Anchor z 偏差：课程前终局；课程开启后触发 recovery。"""
  condition = bad_anchor_pos_z_only(
    env=env, command_name=command_name, threshold=threshold
  )
  return _pre_recovery_terminate_or_post_recovery_enter(env, condition)


def recovery_or_terminate_bad_anchor_ori(
  env: ManagerBasedRlEnv,
  asset_cfg: "SceneEntityCfg",
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  """Anchor 姿态偏差：课程前终局；课程开启后触发 recovery。"""
  condition = bad_anchor_ori(
    env=env, asset_cfg=asset_cfg, command_name=command_name, threshold=threshold
  )
  return _pre_recovery_terminate_or_post_recovery_enter(env, condition)


def recovery_or_terminate_bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """末端等 body z：课程前终局；课程开启后触发 recovery。"""
  condition = bad_motion_body_pos_z_only(
    env=env,
    command_name=command_name,
    threshold=threshold,
    body_names=body_names,
  )
  return _pre_recovery_terminate_or_post_recovery_enter(env, condition)


def recovery_or_terminate_bad_torso_z_vs_motion(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: "SceneEntityCfg",
  body_name: str,
  threshold: float,
) -> torch.Tensor:
  """躯干（或指定 body）世界 z 与参考差过大：仅在 recovery 课程开启**之后**用于进入 recovery。"""
  condition = bad_torso_z_vs_motion_ref(
    env=env,
    command_name=command_name,
    asset_cfg=asset_cfg,
    body_name=body_name,
    threshold=threshold,
  )
  return _recovery_phase_enter_only(env, condition)


def recovery_mismatch_after_duration(
  env: ManagerBasedRlEnv,
  command_name: str,
  recovery_duration_s: float,
  anchor_pos_threshold: float,
  ee_body_pos_threshold: float,
  body_names: tuple[str, ...] | None = None,
  asset_cfg: "SceneEntityCfg | None" = None,
  mismatch_or: bool = True,
  success_stable_steps: int = 6,
  success_hysteresis_decay: int = 1,
) -> torch.Tensor:
  """Recover with event-triggered success and timeout fallback.

  Logic:
  - If recovery is not enabled: return all-False and keep recovery_mask=0.
  - During recovery, evaluate success each step.
  - If success criteria stay valid for `success_stable_steps`, exit recovery now.
  - If still not successful after `recovery_duration_s`, terminate the episode.

  Recovery 内 mismatch 仍用原先组合：anchor z、anchor 投影重力差、末端等 body 的 z（与进 recovery 的躯干 z 条件无关）。
  """
  _ensure_recovery_state(env)
  env_any = cast(Any, env)

  # Default: never terminate.
  terminate = torch.zeros(env_any.num_envs, device=env_any.device, dtype=torch.bool)

  if not bool(env_any.recovery_enabled):
    return terminate

  if not env_any.recovery_mode_buf.any():
    return terminate

  bad_pos_z = bad_anchor_pos_z_only(
    env=env, command_name=command_name, threshold=anchor_pos_threshold
  )
  if asset_cfg is None:
    raise ValueError("asset_cfg must be provided for recovery_mismatch_after_duration")
  bad_ee_pos_z = bad_motion_body_pos_z_only(
    env=env,
    command_name=command_name,
    threshold=ee_body_pos_threshold,
    body_names=body_names,
  )
  mismatch = (
    bad_pos_z | bad_ee_pos_z
    if mismatch_or
    else (bad_pos_z & bad_ee_pos_z)
  )

  success_now = (~mismatch) & env_any.recovery_mode_buf

  # Temporal hysteresis: increase quickly, decay slowly.
  stable_count = env_any.recovery_stable_count_buf
  stable_count[success_now] += 1
  not_success_recovery = env_any.recovery_mode_buf & (~success_now)
  if not_success_recovery.any():
    stable_count[not_success_recovery] = torch.clamp(
      stable_count[not_success_recovery] - int(max(1, success_hysteresis_decay)),
      min=0,
    )

  elapsed_steps = env_any.episode_length_buf - env_any.recovery_start_step_buf
  elapsed_s = elapsed_steps.to(dtype=torch.float32) * float(env.step_dt)

  success_mask = env_any.recovery_mode_buf & (stable_count >= int(max(1, success_stable_steps)))
  timeout_mask = env_any.recovery_mode_buf & (elapsed_s >= recovery_duration_s) & (~success_mask)
  terminate = timeout_mask

  # End recovery immediately when success event is triggered.
  end_mask = success_mask
  success_count = int(torch.count_nonzero(end_mask).item())
  recovery_success_duration_mean_s = 0.0
  if end_mask.any():
    recovery_success_duration_mean_s = float(elapsed_s[end_mask].mean().item())
    normalized_remaining = torch.clamp(
      (float(recovery_duration_s) - elapsed_s[end_mask]) / max(float(recovery_duration_s), 1e-6),
      min=0.0,
      max=1.0,
    )
    env_any.recovery_success_bonus_buf[end_mask] = normalized_remaining
    env_any.recovery_recovered_once_buf[end_mask] = True
    env_any.recovery_mode_buf[end_mask] = False
    env_any.recovery_start_step_buf[end_mask] = 0
    env_any.recovery_stable_count_buf[end_mask] = 0
  env_any.recovery_success_total_count += success_count

  if timeout_mask.any():
    env_any.recovery_stable_count_buf[timeout_mask] = 0

  # Update mask for the runner reward mixer.
  env_any.extras["recovery_mask"] = env_any.recovery_mode_buf.clone()
  env_any.extras["recovery_success_bonus"] = env_any.recovery_success_bonus_buf.clone()
  log = env_any.extras.setdefault("log", {})
  log["Metrics/recovery_success_count_step"] = float(success_count)
  log["Metrics/recovery_success_duration_mean_s_step"] = float(
    recovery_success_duration_mean_s
  )
  denom = max(float(env_any.recovery_enter_total_count), 1.0)
  log["Metrics/recovery_success_rate_total"] = (
    float(env_any.recovery_success_total_count) / denom
  )

  return terminate
