from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import MotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
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
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
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
      env_any._recovery_last_clear_common_step_counter = env_any.common_step_counter

  # Always expose the mask to the AMP runner (it will gate reward mixing).
  env_any.extras["recovery_mask"] = env_any.recovery_mode_buf.clone()


def _maybe_start_recovery_from_condition(
  env: ManagerBasedRlEnv, condition: torch.Tensor
) -> torch.Tensor:
  """Start recovery for envs matching `condition` and return termination mask.

  - If `env.recovery_enabled` is False, this behaves like the original
    termination condition: return `condition` (episode ends).
  - If enabled, start recovery for envs not already in recovery, but return
    all-False to prevent the episode from ending immediately.
  """
  env_any = cast(Any, env)
  _ensure_recovery_state(env)

  # If recovery is not enabled yet, keep the original behavior.
  if not bool(env_any.recovery_enabled):
    return condition

  recovery_mode = env_any.recovery_mode_buf
  # Start recovery only for envs that are not already recovering.
  enter_mask = condition & ~recovery_mode
  if enter_mask.any():
    env_any.recovery_mode_buf[enter_mask] = True
    env_any.recovery_start_step_buf[enter_mask] = env_any.episode_length_buf[
      enter_mask
    ]

  # During recovery mode, we suppress termination from the original terms.
  return torch.zeros_like(condition)


def recovery_or_terminate_bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  """Anchor z position difference: terminate normally or enter recovery."""
  condition = bad_anchor_pos_z_only(
    env=env, command_name=command_name, threshold=threshold
  )
  return _maybe_start_recovery_from_condition(env, condition)


def recovery_or_terminate_bad_anchor_ori(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  """Anchor orientation difference: terminate normally or enter recovery."""
  condition = bad_anchor_ori(
    env=env, asset_cfg=asset_cfg, command_name=command_name, threshold=threshold
  )
  return _maybe_start_recovery_from_condition(env, condition)


def recovery_or_terminate_bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Motion body z pos difference: terminate normally or enter recovery."""
  condition = bad_motion_body_pos_z_only(
    env=env,
    command_name=command_name,
    threshold=threshold,
    body_names=body_names,
  )
  return _maybe_start_recovery_from_condition(env, condition)


def recovery_mismatch_after_duration(
  env: ManagerBasedRlEnv,
  command_name: str,
  recovery_duration_s: float,
  anchor_pos_threshold: float,
  anchor_ori_threshold: float,
  ee_body_pos_threshold: float,
  body_names: tuple[str, ...] | None = None,
  asset_cfg: SceneEntityCfg | None = None,
  mismatch_or: bool = True,
) -> torch.Tensor:
  """Recover for a fixed duration, then decide termination/end recovery.

  Logic:
  - If recovery is not enabled: return all-False and keep recovery_mask=0.
  - If in recovery mode and duration not reached: return all-False.
  - When duration is reached:
      - If mismatch is still large => trigger episode termination.
      - Else => end recovery (unfreeze command & allow mimic reward).
  """
  _ensure_recovery_state(env)
  env_any = cast(Any, env)

  # Default: never terminate.
  terminate = torch.zeros(env_any.num_envs, device=env_any.device, dtype=torch.bool)

  if not bool(env_any.recovery_enabled):
    return terminate

  if not env_any.recovery_mode_buf.any():
    return terminate

  # Compute mismatch based on frozen command reference.
  command = env_any.command_manager.get_term(command_name)
  assert isinstance(command, MotionCommand)

  # Use the same underlying “bad_*” criteria but do not modify recovery state.
  bad_pos_z = bad_anchor_pos_z_only(
    env=env, command_name=command_name, threshold=anchor_pos_threshold
  )
  if asset_cfg is None:
    raise ValueError("asset_cfg must be provided for recovery_mismatch_after_duration")
  bad_ori = bad_anchor_ori(
    env=env,
    asset_cfg=asset_cfg,
    command_name=command_name,
    threshold=anchor_ori_threshold,
  )
  bad_ee_pos_z = bad_motion_body_pos_z_only(
    env=env,
    command_name=command_name,
    threshold=ee_body_pos_threshold,
    body_names=body_names,
  )

  mismatch = bad_pos_z | bad_ori | bad_ee_pos_z if mismatch_or else (bad_pos_z & bad_ori & bad_ee_pos_z)

  elapsed_steps = env_any.episode_length_buf - env_any.recovery_start_step_buf
  elapsed_s = elapsed_steps.to(dtype=torch.float32) * float(env.step_dt)

  ready = env_any.recovery_mode_buf & (elapsed_s >= recovery_duration_s)
  terminate = ready & mismatch

  # End recovery when ready and mismatch is acceptable.
  end_mask = ready & ~mismatch
  if end_mask.any():
    env_any.recovery_mode_buf[end_mask] = False
    env_any.recovery_start_step_buf[end_mask] = 0

  # Update mask for the runner reward mixer.
  env_any.extras["recovery_mask"] = env_any.recovery_mode_buf.clone()

  return terminate
