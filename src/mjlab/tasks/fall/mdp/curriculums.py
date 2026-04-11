from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class PushStage(TypedDict):
  step: int
  x: tuple[float, float]
  y: tuple[float, float]
  z: tuple[float, float]
  roll: tuple[float, float]
  pitch: tuple[float, float]
  yaw: tuple[float, float]


class ResetInitStage(TypedDict):
  step: int
  data_probability: float
  tilt_pose_range: dict[str, tuple[float, float]]
  tilt_velocity_range: dict[str, tuple[float, float]]
  tilt_joint_position_range: tuple[float, float]
  tilt_joint_velocity_range: tuple[float, float]


class ForcePulseStage(TypedDict):
  step: int
  duration_steps_range: tuple[int, int]
  force_axis_range: dict[str, tuple[float, float]]
  torque_axis_range: dict[str, tuple[float, float]]


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
    velocity_range[key] = active_stage[key]

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
  params["tilt_pose_range"] = dict(active_stage["tilt_pose_range"])
  params["tilt_velocity_range"] = dict(active_stage["tilt_velocity_range"])
  params["tilt_joint_position_range"] = active_stage["tilt_joint_position_range"]
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
  """Update reset force pulse magnitude/duration by training stage."""
  del env_ids
  event_term_cfg = env.event_manager.get_term_cfg(event_name)
  params = event_term_cfg.params

  active_stage = pulse_stages[0]
  for stage in pulse_stages:
    if env.common_step_counter >= stage["step"]:
      active_stage = stage

  duration_low_raw, duration_high_raw = active_stage["duration_steps_range"]
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
  params["force_axis_range"] = dict(active_stage["force_axis_range"])
  params["torque_axis_range"] = dict(active_stage["torque_axis_range"])

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


def reset_upward_velocity_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  reward_term_name: str,
  threshold_ratio: float = 0.6,
  decrement: float = 0.02,
  min_upward_velocity: float = 0.0,
) -> dict[str, torch.Tensor]:
  """Adaptively decay reset upward velocity by reward performance.

  Similar to legged-lab get-up `force_level`: when reward performance is good enough,
  reduce reset assistance for the selected envs.
  """
  if env_ids.numel() == 0:
    return {}

  event_term_cfg = env.event_manager.get_term_cfg(event_name)
  velocity_range = event_term_cfg.params["velocity_range"]

  episode_sums = env.reward_manager._episode_sums[reward_term_name]
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
  mean_episode_value = torch.mean(episode_sums[env_ids]) / env.max_episode_length_s
  threshold = float(threshold_ratio) * float(reward_term_cfg.weight)

  current_upward = float(velocity_range.get("z", (0.0, 0.0))[1])
  if float(mean_episode_value.item()) > threshold:
    next_upward = max(float(min_upward_velocity), current_upward - float(decrement))
  else:
    next_upward = current_upward

  # Keep deterministic upward assist at reset.
  velocity_range["z"] = (next_upward, next_upward)

  return {
    "reset_upward_velocity": torch.tensor(next_upward, dtype=torch.float32),
    "reset_upward_threshold": torch.tensor(threshold, dtype=torch.float32),
    "reset_upward_perf": mean_episode_value.detach().to(torch.float32),
  }
