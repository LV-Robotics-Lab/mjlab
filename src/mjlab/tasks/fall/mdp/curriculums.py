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


def reset_push_and_freeze_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  push_stages: list[PushStage],
  short_freeze_range: tuple[int, int],
  long_freeze_range: tuple[int, int],
  long_freeze_ratio: float = 0.5,
) -> dict[str, torch.Tensor]:
  """Update reset push strength by stage and sample per-env freeze durations.

  A subset of reset envs is assigned to a longer freeze range while the rest are
  sampled from a shorter range. This exposes the policy to a mixture of early and
  late takeover timings without manually crafting invalid poses.
  """
  event_term_cfg = env.event_manager.get_term_cfg(event_name)
  velocity_range = event_term_cfg.params["velocity_range"]

  active_stage = push_stages[0]
  for stage in push_stages:
    if env.common_step_counter >= stage["step"]:
      active_stage = stage

  for key in ("x", "y", "z", "roll", "pitch", "yaw"):
    velocity_range[key] = active_stage[key]

  if isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  if env_ids.numel() == 0:
    return {
      "freeze_short_min": torch.tensor(short_freeze_range[0], dtype=torch.float32),
      "freeze_short_max": torch.tensor(short_freeze_range[1], dtype=torch.float32),
      "freeze_long_min": torch.tensor(long_freeze_range[0], dtype=torch.float32),
      "freeze_long_max": torch.tensor(long_freeze_range[1], dtype=torch.float32),
      "freeze_long_ratio": torch.tensor(long_freeze_ratio, dtype=torch.float32),
      "freeze_mean": torch.tensor(0.0, dtype=torch.float32),
      "push_x_max": torch.tensor(abs(velocity_range["x"][1]), dtype=torch.float32),
      "push_pitch_max": torch.tensor(abs(velocity_range["pitch"][1]), dtype=torch.float32),
    }

  num_reset = env_ids.numel()
  num_long = int(round(num_reset * long_freeze_ratio))
  perm = torch.randperm(num_reset, device=env.device)
  long_ids = env_ids[perm[:num_long]]
  short_ids = env_ids[perm[num_long:]]

  freeze_buf = env.post_reset_freeze_steps_buf
  short_low, short_high = short_freeze_range
  long_low, long_high = long_freeze_range

  if short_ids.numel() > 0:
    freeze_buf[short_ids] = torch.randint(
      short_low,
      short_high + 1,
      (short_ids.numel(),),
      device=env.device,
      dtype=torch.long,
    )
  if long_ids.numel() > 0:
    freeze_buf[long_ids] = torch.randint(
      long_low,
      long_high + 1,
      (long_ids.numel(),),
      device=env.device,
      dtype=torch.long,
    )

  return {
    "freeze_short_min": torch.tensor(short_low, dtype=torch.float32),
    "freeze_short_max": torch.tensor(short_high, dtype=torch.float32),
    "freeze_long_min": torch.tensor(long_low, dtype=torch.float32),
    "freeze_long_max": torch.tensor(long_high, dtype=torch.float32),
    "freeze_long_ratio": torch.tensor(long_freeze_ratio, dtype=torch.float32),
    "freeze_mean": freeze_buf[env_ids].float().mean(),
    "push_x_max": torch.tensor(abs(velocity_range["x"][1]), dtype=torch.float32),
    "push_pitch_max": torch.tensor(abs(velocity_range["pitch"][1]), dtype=torch.float32),
  }
