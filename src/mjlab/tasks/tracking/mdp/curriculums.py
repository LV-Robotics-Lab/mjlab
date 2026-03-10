"""Curriculum terms for tracking task."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# 轨迹文件名 → 初速度锥形中心角（度，0=正前方+x）
MOTION_DIRECTION_DEG = {
  "front": 0.0,
  "back": 180.0,
  "left": -90.0,
  "right": 90.0,
  "leftfront": -45.0,
  "rightfront": 45.0,
  "leftback": -135.0,
  "rightback": 135.0,
}


def _infer_cone_center_deg_from_motion_file(motion_file: str) -> float:
  """从 motion 文件路径/文件名推断方向，返回锥形中心角（度）。未匹配则 0（正前方）。"""
  if not motion_file:
    return 0.0
  stem = Path(motion_file).stem.lower()
  # 先匹配复合方向，再匹配单方向
  for key in ("leftfront", "rightfront", "leftback", "rightback", "front", "back", "left", "right"):
    if key in stem:
      return MOTION_DIRECTION_DEG[key]
  return 0.0


class ScaleStage(TypedDict):
  step: int
  scale: float


def initial_velocity_range(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  scale_stages: list[ScaleStage],
) -> dict[str, torch.Tensor]:
  """随训练步数将 initial velocity 从 0 逐渐放大到 velocity_range（各轴独立）。"""
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
  env.initial_velocity_forward = None
  out = {"scale": torch.tensor(scale, device=env.device)}
  for k, v in scaled.items():
    out[f"{k}_min"] = torch.tensor(v[0], device=env.device)
    out[f"{k}_max"] = torch.tensor(v[1], device=env.device)
  return out


def initial_velocity_forward(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  max_speed: float,
  scale_stages: list[ScaleStage],
  angle_range_deg: tuple[float, float] | None = None,
  cone_half_deg: float = 22.5,
  z: tuple[float, float] = (0.0, 0.0),
  roll: tuple[float, float] = (0.0, 0.0),
  pitch: tuple[float, float] = (0.0, 0.0),
  yaw: tuple[float, float] = (0.0, 0.0),
) -> dict[str, torch.Tensor]:
  """初速度限制在「当前轨迹方向 ±cone_half_deg」锥形内；方向由 motion 文件名自动推断（Front/Back/Left/Right/LeftFront 等）。"""
  del env_ids
  # 从 motion 文件路径推断锥形中心角
  motion_file = ""
  if env.command_manager is not None and "motion" in env.command_manager.active_terms:
    cmd = env.command_manager.get_term("motion")
    motion_file = getattr(cmd.cfg, "motion_file", "") or ""
  center_deg = _infer_cone_center_deg_from_motion_file(motion_file)
  if angle_range_deg is not None:
    lo, hi = angle_range_deg
  else:
    lo, hi = center_deg - cone_half_deg, center_deg + cone_half_deg
  scale = 0.0
  for stage in scale_stages:
    if env.common_step_counter >= stage["step"]:
      scale = stage["scale"]
  scaled_max = scale * max_speed
  env.initial_velocity_range = None
  env.initial_velocity_forward = {
    "speed_range": (0.0, scaled_max),
    "angle_range_deg": (lo, hi),
    "z": z,
    "roll": roll,
    "pitch": pitch,
    "yaw": yaw,
  }
  return {
    "scale": torch.tensor(scale, device=env.device),
    "speed_max": torch.tensor(scaled_max, device=env.device),
    "cone_center_deg": torch.tensor(center_deg, device=env.device),
  }
