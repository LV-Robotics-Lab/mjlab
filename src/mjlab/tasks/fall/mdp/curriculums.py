from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from typing_extensions import NotRequired

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class PushStage(TypedDict):
  step: int
  x: NotRequired[tuple[float, float]]
  y: NotRequired[tuple[float, float]]
  z: NotRequired[tuple[float, float]]
  roll: NotRequired[tuple[float, float]]
  pitch: NotRequired[tuple[float, float]]
  yaw: NotRequired[tuple[float, float]]


class ResetInitStage(TypedDict):
  step: int
  data_probability: float
  tilt_pose_range: NotRequired[dict[str, tuple[float, float]]]
  tilt_velocity_range: NotRequired[dict[str, tuple[float, float]]]
  tilt_joint_position_range: NotRequired[tuple[float, float]]
  tilt_joint_velocity_range: NotRequired[tuple[float, float]]


class ForcePulseStage(TypedDict):
  step: int
  duration_steps_range: NotRequired[tuple[int, int]]
  force_axis_range: NotRequired[dict[str, tuple[float, float]]]
  torque_axis_range: NotRequired[dict[str, tuple[float, float]]]


def _lerp(a: float, b: float, t: float) -> float:
  return (1.0 - t) * a + t * b


def _lerp_range(
  lhs: tuple[float, float], rhs: tuple[float, float], t: float
) -> tuple[float, float]:
  return (_lerp(lhs[0], rhs[0], t), _lerp(lhs[1], rhs[1], t))


def _lerp_axis_ranges(
  current: dict[str, tuple[float, float]],
  nxt: dict[str, tuple[float, float]],
  t: float,
) -> dict[str, tuple[float, float]]:
  result = dict(current)
  for key, cur_range in current.items():
    next_range = nxt.get(key)
    if next_range is None:
      continue
    result[key] = _lerp_range(cur_range, next_range, t)
  return result


def reset_push_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  push_stages: list[PushStage],
) -> dict[str, torch.Tensor]:
  """Update reset push strength by training stage."""
  event_term_cfg = env.event_manager.get_term_cfg(event_name)
  velocity_range = event_term_cfg.params["velocity_range"]

  active_stage = push_stages[0]
  for stage in push_stages:
    if env.common_step_counter >= stage["step"]:
      active_stage = stage

  for key in ("x", "y", "z", "roll", "pitch", "yaw"):
    stage_range = active_stage.get(key)
    if stage_range is not None:
      velocity_range[key] = stage_range

  return {
    "push_x_max": torch.tensor(abs(velocity_range["x"][1]), dtype=torch.float32),
    "push_y_max": torch.tensor(abs(velocity_range["y"][1]), dtype=torch.float32),
    "push_pitch_max": torch.tensor(abs(velocity_range["pitch"][1]), dtype=torch.float32),
    "push_roll_max": torch.tensor(abs(velocity_range["roll"][1]), dtype=torch.float32),
  }


def reset_initialization_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  init_stages: list[ResetInitStage],
) -> dict[str, torch.Tensor]:
  """Update reset initialization difficulty and data mixing by training stage."""
  del env_ids
  event_term_cfg = env.event_manager.get_term_cfg(event_name)
  params = event_term_cfg.params

  active_stage = init_stages[0]
  for stage in init_stages:
    if env.common_step_counter >= stage["step"]:
      active_stage = stage

  params["data_probability"] = active_stage["data_probability"]
  if "tilt_pose_range" in active_stage:
    params["tilt_pose_range"] = dict(active_stage["tilt_pose_range"])
  if "tilt_velocity_range" in active_stage:
    params["tilt_velocity_range"] = dict(active_stage["tilt_velocity_range"])
  if "tilt_joint_position_range" in active_stage:
    params["tilt_joint_position_range"] = active_stage["tilt_joint_position_range"]
  if "tilt_joint_velocity_range" in active_stage:
    params["tilt_joint_velocity_range"] = active_stage["tilt_joint_velocity_range"]

  tilt_pose_range = params["tilt_pose_range"]
  return {
    "reset_data_probability": torch.tensor(
      params["data_probability"], dtype=torch.float32
    ),
    "reset_tilt_roll_max": torch.tensor(
      max(abs(v) for v in tilt_pose_range.get("roll", (0.0, 0.0))),
      dtype=torch.float32,
    ),
    "reset_tilt_pitch_max": torch.tensor(
      max(abs(v) for v in tilt_pose_range.get("pitch", (0.0, 0.0))),
      dtype=torch.float32,
    ),
  }


def reset_force_pulse_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  pulse_stages: list[ForcePulseStage],
) -> dict[str, torch.Tensor]:
  """Update reset force pulse magnitude/duration by training stage.

  To avoid sharp distribution shifts at stage boundaries, this curriculum
  linearly interpolates between adjacent stages.
  """
  del env_ids
  event_term_cfg = env.event_manager.get_term_cfg(event_name)
  params = event_term_cfg.params

  active_idx = 0
  for i, stage in enumerate(pulse_stages):
    if env.common_step_counter >= stage["step"]:
      active_idx = i
  active_stage = pulse_stages[active_idx]
  next_stage = pulse_stages[min(active_idx + 1, len(pulse_stages) - 1)]
  if next_stage["step"] == active_stage["step"]:
    blend = 0.0
  else:
    blend = float(
      (env.common_step_counter - active_stage["step"])
      / (next_stage["step"] - active_stage["step"])
    )
    blend = max(0.0, min(1.0, blend))

  active_duration = active_stage.get("duration_steps_range", params["duration_steps_range"])
  next_duration = next_stage.get("duration_steps_range", active_duration)
  duration_steps_range_float = _lerp_range(
    (float(active_duration[0]), float(active_duration[1])),
    (float(next_duration[0]), float(next_duration[1])),
    blend,
  )
  duration_steps_range = (
    int(round(min(duration_steps_range_float))),
    int(round(max(duration_steps_range_float))),
  )
  duration_low_raw, duration_high_raw = duration_steps_range
  duration_low = int(min(duration_low_raw, duration_high_raw))
  duration_high = int(max(duration_low_raw, duration_high_raw))
  sampled_duration = int(
    torch.randint(
      low=duration_low,
      high=duration_high + 1,
      size=(1,),
      device=env.device,
    ).item()
  )
  params["duration_steps"] = sampled_duration
  active_force = active_stage.get("force_axis_range", params["force_axis_range"])
  next_force = next_stage.get("force_axis_range", active_force)
  params["force_axis_range"] = _lerp_axis_ranges(active_force, next_force, blend)
  active_torque = active_stage.get("torque_axis_range", params["torque_axis_range"])
  next_torque = next_stage.get("torque_axis_range", active_torque)
  params["torque_axis_range"] = _lerp_axis_ranges(active_torque, next_torque, blend)

  force_axis_range = params["force_axis_range"]
  torque_axis_range = params["torque_axis_range"]
  return {
    "force_pulse_duration_steps": torch.tensor(
      params["duration_steps"], dtype=torch.float32
    ),
    "force_pulse_abs_fx_max": torch.tensor(
      max(abs(v) for v in force_axis_range.get("x", (0.0, 0.0))),
      dtype=torch.float32,
    ),
    "force_pulse_abs_fy_max": torch.tensor(
      max(abs(v) for v in force_axis_range.get("y", (0.0, 0.0))),
      dtype=torch.float32,
    ),
    "force_pulse_abs_fz_max": torch.tensor(
      max(abs(v) for v in force_axis_range.get("z", (0.0, 0.0))),
      dtype=torch.float32,
    ),
    "force_pulse_abs_pitch_torque_max": torch.tensor(
      max(abs(v) for v in torque_axis_range.get("pitch", (0.0, 0.0))),
      dtype=torch.float32,
    ),
  }
