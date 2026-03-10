"""Curriculum terms for tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class ScaleStage(TypedDict):
  step: int
  scale: float


def initial_velocity_range(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  scale_stages: list[ScaleStage],
) -> dict[str, torch.Tensor]:
  """随训练步数将 initial velocity 从 0 逐渐放大到 velocity_range。

  使用 scale_stages 中 step <= common_step_counter 的最后一个 scale，
  对 velocity_range 各轴做缩放：current_range[k] = (scale * v[0], scale * v[1])。
  """
  del env_ids
  scale = 0.0
  for stage in scale_stages:
    if env.common_step_counter >= stage["step"]:
      scale = stage["scale"]
  scaled = {
    k: (scale * v[0], scale * v[1])
    for k, v in velocity_range.items()
  }
  env.initial_velocity_range = scaled
  out = {"scale": torch.tensor(scale, device=env.device)}
  for k, v in scaled.items():
    out[f"{k}_min"] = torch.tensor(v[0], device=env.device)
    out[f"{k}_max"] = torch.tensor(v[1], device=env.device)
  return out
