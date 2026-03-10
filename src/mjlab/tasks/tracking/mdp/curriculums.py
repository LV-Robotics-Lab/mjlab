"""Curriculum terms for tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class InitialVelocityStage(TypedDict, total=False):
  step: int
  x: tuple[float, float]
  y: tuple[float, float]
  z: tuple[float, float]
  roll: tuple[float, float]
  pitch: tuple[float, float]
  yaw: tuple[float, float]


def initial_velocity_range(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_stages: list[InitialVelocityStage],
) -> dict[str, torch.Tensor]:
  """Set env.initial_velocity_range from velocity_stages based on common_step_counter.

  Uses the last stage whose step <= env.common_step_counter.
  """
  del env_ids
  velocity_range = {}
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]:
        if key in stage and stage[key] is not None:
          velocity_range[key] = stage[key]
  env.initial_velocity_range = velocity_range if velocity_range else None
  out = {}
  for k, v in (velocity_range or {}).items():
    out[f"{k}_min"] = torch.tensor(v[0], device=env.device)
    out[f"{k}_max"] = torch.tensor(v[1], device=env.device)
  return out
