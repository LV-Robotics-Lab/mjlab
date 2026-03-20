from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_MOTION_RESET_CACHE: dict[tuple[str, int, str], dict[str, torch.Tensor]] = {}
_LAST_RESET_NPZ_MASK_ATTR = "_fall_last_reset_npz_mask"


def _load_motion_reset_file(
  path: str,
  root_body_idx: int,
  device: str,
  expected_num_joints: int,
) -> dict[str, torch.Tensor]:
  cache_key = (path, root_body_idx, device)
  cached = _MOTION_RESET_CACHE.get(cache_key)
  if cached is not None:
    return cached

  data = np.load(path)
  joint_pos = torch.as_tensor(data["joint_pos"], dtype=torch.float32, device=device)
  joint_vel = torch.as_tensor(data["joint_vel"], dtype=torch.float32, device=device)
  if joint_pos.ndim != 2 or joint_vel.ndim != 2:
    raise ValueError(
      f"Reset motion '{path}' must store 2D joint arrays, got "
      f"joint_pos={joint_pos.shape}, joint_vel={joint_vel.shape}."
    )
  if joint_pos.shape != joint_vel.shape:
    raise ValueError(
      f"Reset motion '{path}' has mismatched joint shapes: "
      f"{joint_pos.shape} vs {joint_vel.shape}."
    )
  if joint_pos.shape[1] != expected_num_joints:
    raise ValueError(
      f"Reset motion '{path}' joint count {joint_pos.shape[1]} does not match "
      f"robot joint count {expected_num_joints}."
    )

  if "body_pos_w" not in data or "body_quat_w" not in data:
    raise ValueError(
      f"Reset motion '{path}' must contain tracking-style body arrays "
      "('body_pos_w', 'body_quat_w')."
    )

  body_pos = np.asarray(data["body_pos_w"])
  body_quat = np.asarray(data["body_quat_w"])
  if body_pos.ndim == 2:
    body_pos = body_pos[:, np.newaxis, :]
    body_quat = body_quat[:, np.newaxis, :]
  if root_body_idx >= body_pos.shape[1]:
    raise ValueError(
      f"Reset motion '{path}' has only {body_pos.shape[1]} bodies, "
      f"cannot index root body {root_body_idx}."
    )

  if "body_lin_vel_w" in data and "body_ang_vel_w" in data:
    body_lin_vel = np.asarray(data["body_lin_vel_w"])
    body_ang_vel = np.asarray(data["body_ang_vel_w"])
    if body_lin_vel.ndim == 2:
      body_lin_vel = body_lin_vel[:, np.newaxis, :]
      body_ang_vel = body_ang_vel[:, np.newaxis, :]
  else:
    zeros = np.zeros_like(body_pos)
    body_lin_vel = zeros
    body_ang_vel = zeros

  root_state = torch.cat(
    [
      torch.as_tensor(body_pos[:, root_body_idx], dtype=torch.float32, device=device),
      torch.as_tensor(body_quat[:, root_body_idx], dtype=torch.float32, device=device),
      torch.as_tensor(body_lin_vel[:, root_body_idx], dtype=torch.float32, device=device),
      torch.as_tensor(body_ang_vel[:, root_body_idx], dtype=torch.float32, device=device),
    ],
    dim=-1,
  )
  cached = {
    "root_state": root_state,
    "joint_pos": joint_pos,
    "joint_vel": joint_vel,
  }
  _MOTION_RESET_CACHE[cache_key] = cached
  return cached


def _default_states(
  env: ManagerBasedRlEnv,
  asset: Entity,
  env_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  default_root_state = asset.data.default_root_state
  default_joint_pos = asset.data.default_joint_pos
  default_joint_vel = asset.data.default_joint_vel
  assert default_root_state is not None
  assert default_joint_pos is not None
  assert default_joint_vel is not None

  root_state = default_root_state[env_ids].clone()
  root_state[:, 0:3] += env.scene.env_origins[env_ids]
  joint_pos = default_joint_pos[env_ids].clone()
  joint_vel = default_joint_vel[env_ids].clone()
  return root_state, joint_pos, joint_vel


def _sample_tilt_states(
  env: ManagerBasedRlEnv,
  asset: Entity,
  env_ids: torch.Tensor,
  tilt_pose_range: dict[str, tuple[float, float]],
  tilt_velocity_range: dict[str, tuple[float, float]] | None,
  tilt_joint_position_range: tuple[float, float],
  tilt_joint_velocity_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  root_state, joint_pos, joint_vel = _default_states(env, asset, env_ids)

  pose_ranges = torch.tensor(
    [
      tilt_pose_range.get(key, (0.0, 0.0))
      for key in ("x", "y", "z", "roll", "pitch", "yaw")
    ],
    device=env.device,
    dtype=torch.float32,
  )
  pose_samples = sample_uniform(
    pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=env.device
  )
  root_state[:, 0:3] += pose_samples[:, 0:3]
  root_state[:, 3:7] = quat_mul(
    root_state[:, 3:7],
    quat_from_euler_xyz(
      pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    ),
  )

  if tilt_velocity_range is None:
    tilt_velocity_range = {}
  vel_ranges = torch.tensor(
    [
      tilt_velocity_range.get(key, (0.0, 0.0))
      for key in ("x", "y", "z", "roll", "pitch", "yaw")
    ],
    device=env.device,
    dtype=torch.float32,
  )
  vel_samples = sample_uniform(
    vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=env.device
  )
  root_state[:, 7:13] += vel_samples

  joint_pos += sample_uniform(
    tilt_joint_position_range[0],
    tilt_joint_position_range[1],
    joint_pos.shape,
    device=env.device,
  )
  joint_vel += sample_uniform(
    tilt_joint_velocity_range[0],
    tilt_joint_velocity_range[1],
    joint_vel.shape,
    device=env.device,
  )

  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None
  joint_limits = soft_joint_pos_limits[env_ids]
  joint_pos = joint_pos.clamp_(joint_limits[..., 0], joint_limits[..., 1])

  return root_state, joint_pos, joint_vel


def _sample_motion_states(
  env: ManagerBasedRlEnv,
  asset: Entity,
  env_ids: torch.Tensor,
  motion_files: Sequence[str],
  npz_root_body_name: str,
  npz_frame_range: tuple[float, float],
  asset_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  root_ids, _ = asset.find_bodies((npz_root_body_name,), preserve_order=True)
  if not root_ids:
    raise ValueError(
      f"Could not find npz root body '{npz_root_body_name}' in asset "
      f"'{asset_name}'."
    )
  root_body_idx = root_ids[0]

  datasets = [
    _load_motion_reset_file(
      path=path,
      root_body_idx=root_body_idx,
      device=env.device,
      expected_num_joints=asset.num_joints,
    )
    for path in motion_files
  ]
  if not datasets:
    raise ValueError("npz reset requested but no motion files were configured.")

  low_frac = float(np.clip(npz_frame_range[0], 0.0, 1.0))
  high_frac = float(np.clip(npz_frame_range[1], low_frac, 1.0))

  root_state = torch.zeros((len(env_ids), 13), device=env.device)
  joint_pos = torch.zeros((len(env_ids), asset.num_joints), device=env.device)
  joint_vel = torch.zeros((len(env_ids), asset.num_joints), device=env.device)
  motion_ids = torch.randint(
    low=0,
    high=len(datasets),
    size=(len(env_ids),),
    device=env.device,
  )

  for motion_idx, motion in enumerate(datasets):
    local_ids = torch.nonzero(motion_ids == motion_idx, as_tuple=False).squeeze(-1)
    if local_ids.numel() == 0:
      continue

    num_frames = motion["root_state"].shape[0]
    start_idx = min(int(low_frac * max(num_frames - 1, 0)), num_frames - 1)
    end_idx = min(int(high_frac * max(num_frames - 1, 0)), num_frames - 1)
    end_idx = max(start_idx, end_idx)
    frame_ids = torch.randint(
      low=start_idx,
      high=end_idx + 1,
      size=(local_ids.numel(),),
      device=env.device,
    )
    root_state[local_ids] = motion["root_state"][frame_ids]
    root_state[local_ids, 0:3] += env.scene.env_origins[env_ids[local_ids]]
    joint_pos[local_ids] = motion["joint_pos"][frame_ids]
    joint_vel[local_ids] = motion["joint_vel"][frame_ids]

  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None
  joint_limits = soft_joint_pos_limits[env_ids]
  joint_pos = joint_pos.clamp_(joint_limits[..., 0], joint_limits[..., 1])
  return root_state, joint_pos, joint_vel


def _write_state_and_forward(
  env: ManagerBasedRlEnv,
  asset: Entity,
  env_ids: torch.Tensor,
  root_state: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
) -> None:
  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
  asset.write_root_state_to_sim(root_state, env_ids=env_ids)
  asset.clear_state(env_ids=env_ids)
  env.sim.forward()


def reset_root_state_mixed(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  tilt_pose_range: dict[str, tuple[float, float]],
  tilt_velocity_range: dict[str, tuple[float, float]] | None = None,
  tilt_joint_position_range: tuple[float, float] = (0.0, 0.0),
  tilt_joint_velocity_range: tuple[float, float] = (0.0, 0.0),
  npz_probability: float = 0.0,
  motion_files: Sequence[str] = (),
  npz_frame_range: tuple[float, float] = (0.0, 1.0),
  npz_root_body_name: str = "LINK_BASE",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset robot using a mixture of dangerous tilt states and motion npz states."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return

  reset_npz_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  setattr(env, _LAST_RESET_NPZ_MASK_ATTR, reset_npz_mask)

  asset: Entity = env.scene[asset_cfg.name]
  use_npz = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
  if motion_files and npz_probability > 0.0:
    use_npz = torch.rand(len(env_ids), device=env.device) < float(npz_probability)

  tilt_env_ids = env_ids[~use_npz]
  if tilt_env_ids.numel() > 0:
    tilt_root, tilt_joint_pos, tilt_joint_vel = _sample_tilt_states(
      env=env,
      asset=asset,
      env_ids=tilt_env_ids,
      tilt_pose_range=tilt_pose_range,
      tilt_velocity_range=tilt_velocity_range,
      tilt_joint_position_range=tilt_joint_position_range,
      tilt_joint_velocity_range=tilt_joint_velocity_range,
    )
    _write_state_and_forward(
      env, asset, tilt_env_ids, tilt_root, tilt_joint_pos, tilt_joint_vel
    )

  npz_env_ids = env_ids[use_npz]
  if npz_env_ids.numel() > 0:
    reset_npz_mask[npz_env_ids] = True
    npz_root, npz_joint_pos, npz_joint_vel = _sample_motion_states(
      env=env,
      asset=asset,
      env_ids=npz_env_ids,
      motion_files=motion_files,
      npz_root_body_name=npz_root_body_name,
      npz_frame_range=npz_frame_range,
      asset_name=asset_cfg.name,
    )
    _write_state_and_forward(
      env, asset, npz_env_ids, npz_root, npz_joint_pos, npz_joint_vel
    )


def push_by_setting_velocity_preserve_npz(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  preserve_npz_reset_states: bool = True,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Apply reset push without modifying envs initialized from npz motions."""
  if preserve_npz_reset_states:
    npz_mask = getattr(env, _LAST_RESET_NPZ_MASK_ATTR, None)
    if npz_mask is not None:
      npz_mask = npz_mask.to(env.device).bool()
      env_ids = env_ids[~npz_mask[env_ids]]
  if env_ids.numel() == 0:
    return

  asset: Entity = env.scene[asset_cfg.name]
  vel_w = asset.data.root_link_vel_w[env_ids]
  range_list = [
    velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  vel_w += sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)
  asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)
