from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def tracking_recovery_curriculum(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  *,
  recovery_start_common_step: int,
) -> dict[str, torch.Tensor]:
  """Only toggle recovery-mode enable flag by training step."""
  del env_ids  # CurriculumManager passes env_ids, but this curriculum is global.
  env_any = cast(Any, env)
  recovery_enabled = env_any.common_step_counter >= recovery_start_common_step
  env_any.recovery_enabled = bool(recovery_enabled)
  return {
    "recovery_enabled": torch.tensor(float(recovery_enabled), device=env_any.device),
    "common_step_counter": torch.tensor(
      float(env_any.common_step_counter), device=env_any.device
    ),
  }


def tracking_push_force_curriculum(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  *,
  event_name: str,
  recovery_start_common_step: int,
  recovery_force_axis_range: dict[str, tuple[float, float]],
  recovery_torque_axis_range: dict[str, tuple[float, float]],
  recovery_duration_steps_range: tuple[int, int],
) -> dict[str, torch.Tensor]:
  """Set push-force pulse ranges by recovery stage (no scaling)."""
  del env_ids
  env_any = cast(Any, env)
  event_term_cfg = env_any.event_manager.get_term_cfg(event_name)
  params = event_term_cfg.params

  if not hasattr(env_any, "_tracking_base_force_axis_range"):
    env_any._tracking_base_force_axis_range = dict(params["force_axis_range"])
  if not hasattr(env_any, "_tracking_base_torque_axis_range"):
    env_any._tracking_base_torque_axis_range = dict(params.get("torque_axis_range", {}))
  if not hasattr(env_any, "_tracking_base_duration_steps_range"):
    env_any._tracking_base_duration_steps_range = tuple(
      params.get("duration_steps_range", (1, 1))
    )

  recovery_enabled = env_any.common_step_counter >= recovery_start_common_step
  if recovery_enabled:
    active_force_axis_range = dict(recovery_force_axis_range)
    active_torque_axis_range = dict(recovery_torque_axis_range)
    duration_low_raw, duration_high_raw = recovery_duration_steps_range
  else:
    active_force_axis_range = dict(env_any._tracking_base_force_axis_range)
    active_torque_axis_range = dict(env_any._tracking_base_torque_axis_range)
    duration_low_raw, duration_high_raw = env_any._tracking_base_duration_steps_range

  duration_low = int(min(duration_low_raw, duration_high_raw))
  duration_high = int(max(duration_low_raw, duration_high_raw))
  sampled_duration = int(
    torch.randint(
      low=duration_low,
      high=duration_high + 1,
      size=(1,),
      device=env_any.device,
    ).item()
  )

  params["force_axis_range"] = active_force_axis_range
  params["torque_axis_range"] = active_torque_axis_range
  params["duration_steps"] = sampled_duration

  return {
    "recovery_enabled": torch.tensor(float(recovery_enabled), device=env_any.device),
    "force_pulse_duration_steps": torch.tensor(sampled_duration, dtype=torch.float32),
    "force_pulse_abs_fx_max": torch.tensor(
      max(abs(v) for v in active_force_axis_range.get("x", (0.0, 0.0))),
      dtype=torch.float32,
    ),
    "force_pulse_abs_fy_max": torch.tensor(
      max(abs(v) for v in active_force_axis_range.get("y", (0.0, 0.0))),
      dtype=torch.float32,
    ),
    "force_pulse_abs_fz_max": torch.tensor(
      max(abs(v) for v in active_force_axis_range.get("z", (0.0, 0.0))),
      dtype=torch.float32,
    ),
    "force_pulse_abs_pitch_torque_max": torch.tensor(
      max(abs(v) for v in active_torque_axis_range.get("pitch", (0.0, 0.0))),
      dtype=torch.float32,
    ),
    "common_step_counter": torch.tensor(
      float(env_any.common_step_counter), device=env_any.device
    ),
  }


def tracking_recovery_disc_weight_curriculum(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  *,
  stages: list[dict[str, float]],
) -> dict[str, torch.Tensor]:
  """Set recovery-phase discriminator reward scale by training step."""
  del env_ids
  env_any = cast(Any, env)

  if len(stages) == 0:
    scale = 1.0
  else:
    sorted_stages = sorted(stages, key=lambda x: int(x.get("step", 0)))
    scale = float(sorted_stages[0].get("scale", 1.0))
    current_step = int(env_any.common_step_counter)
    for stage in sorted_stages:
      if current_step >= int(stage.get("step", 0)):
        scale = float(stage.get("scale", scale))
      else:
        break

  env_any.recovery_disc_weight_scale = scale
  return {
    "recovery_disc_weight_scale": torch.tensor(scale, device=env_any.device),
    "common_step_counter": torch.tensor(
      float(env_any.common_step_counter), device=env_any.device
    ),
  }


def tracking_recovery_entry_penalty_curriculum(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  *,
  stages: list[dict[str, float]],
) -> dict[str, torch.Tensor]:
  """Set recovery-entry penalty scale by training step."""
  del env_ids
  env_any = cast(Any, env)

  if len(stages) == 0:
    scale = 1.0
  else:
    sorted_stages = sorted(stages, key=lambda x: int(x.get("step", 0)))
    scale = float(sorted_stages[0].get("scale", 1.0))
    current_step = int(env_any.common_step_counter)
    for stage in sorted_stages:
      if current_step >= int(stage.get("step", 0)):
        scale = float(stage.get("scale", scale))
      else:
        break

  env_any.recovery_entry_penalty_scale = scale
  return {
    "recovery_entry_penalty_scale": torch.tensor(scale, device=env_any.device),
    "common_step_counter": torch.tensor(
      float(env_any.common_step_counter), device=env_any.device
    ),
  }


def tracking_recovery_task_weight_curriculum(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  *,
  stages: list[dict[str, float]],
) -> dict[str, torch.Tensor]:
  """Set recovery-phase task reward scale by training step."""
  del env_ids
  env_any = cast(Any, env)

  if len(stages) == 0:
    scale = 1.0
  else:
    sorted_stages = sorted(stages, key=lambda x: int(x.get("step", 0)))
    scale = float(sorted_stages[0].get("scale", 1.0))
    current_step = int(env_any.common_step_counter)
    for stage in sorted_stages:
      if current_step >= int(stage.get("step", 0)):
        scale = float(stage.get("scale", scale))
      else:
        break

  env_any.recovery_task_weight_scale = scale
  return {
    "recovery_task_weight_scale": torch.tensor(scale, device=env_any.device),
    "common_step_counter": torch.tensor(
      float(env_any.common_step_counter), device=env_any.device
    ),
  }

