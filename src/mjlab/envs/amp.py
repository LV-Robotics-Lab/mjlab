"""AMP (Adversarial Motion Priors) support for manager-based RL envs.

Provides disc_obs (discriminator observation) from robot state history,
get_disc_obs_space(), and fetch_disc_obs_demo() for use with AMP-style training
(e.g. MimicKit amp_agent, or amp-rsl-rl).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.utils.buffers.circular_buffer import CircularBuffer
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_mul,
  subtract_frame_transforms,
)
from mjlab.utils.spaces import Box

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _quat_to_6d(quat: torch.Tensor) -> torch.Tensor:
  """Quaternion to 6D rotation (first two columns of R). (..., 4) -> (..., 6)."""
  mat = matrix_from_quat(quat)
  return mat[..., :2].reshape(*quat.shape[:-1], 6)


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
  """Extract yaw (around Z) from quaternion (w, x, y, z)."""
  w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  return torch.atan2(siny_cosp, cosy_cosp)


def _heading_quat_inv(quat: torch.Tensor) -> torch.Tensor:
  """Inverse of yaw-only quaternion. quat (N, 4) wxyz."""
  from mjlab.utils.lab_api.math import quat_conjugate, quat_from_euler_xyz
  yaw = _yaw_from_quat(quat)
  q_yaw_inv = quat_from_euler_xyz(
    torch.zeros_like(yaw), torch.zeros_like(yaw), -yaw
  )
  return quat_conjugate(q_yaw_inv)


def compute_disc_obs(
  ref_root_pos: torch.Tensor,
  ref_root_quat: torch.Tensor,
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  root_lin_vel: torch.Tensor,
  root_ang_vel: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  global_obs: bool = False,
  root_height_obs: bool = True,
  include_root_xy: bool = True,
  include_root_rot: bool = True,
  include_root_vel: bool = True,
  include_projected_gravity: bool = False,
  extra_body_pos_w: torch.Tensor | None = None,
  extra_body_quat_w: torch.Tensor | None = None,
) -> torch.Tensor:
  """Compute discriminator observation from history of states.

  All inputs except ref_* have shape (num_envs, num_steps, ...). ref_* (num_envs, ...).
  Output (num_envs, disc_dim).
  """
  n, t = root_pos.shape[0], root_pos.shape[1]
  ref_pos = ref_root_pos.unsqueeze(1)
  ref_quat = ref_root_quat.unsqueeze(1)

  root_pos_rel = root_pos - ref_pos

  if not global_obs:
    heading_inv = _heading_quat_inv(ref_root_quat)
    heading_inv_expand = heading_inv.unsqueeze(1).expand(n, t, 4)
    root_pos_rel_flat = root_pos_rel.reshape(-1, 3)
    heading_flat = heading_inv_expand.reshape(-1, 4)
    root_pos_rel = quat_apply_inverse(heading_flat, root_pos_rel_flat).reshape(n, t, 3)
    heading_inv_expand = heading_inv.unsqueeze(1).expand(n, t, 4)
    root_quat_local = quat_mul(heading_inv_expand, root_quat)
    if include_root_vel:
      root_lin_vel = quat_apply_inverse(
        heading_flat, root_lin_vel.reshape(-1, 3)
      ).reshape(n, t, 3)
      root_ang_vel = quat_apply_inverse(
        heading_flat, root_ang_vel.reshape(-1, 3)
      ).reshape(n, t, 3)
  else:
    root_quat_local = root_quat

  root_pos_terms: list[torch.Tensor] = []
  if include_root_xy:
    root_pos_terms.append(root_pos_rel[..., :2])
  if root_height_obs:
    # Use actual root height rather than height relative to the reference
    # frame. With num_disc_obs_steps == 1, the relative z would otherwise be
    # identically zero and provide no fall-state information.
    root_pos_terms.append(root_pos[..., 2:3])

  root_rot_6d = _quat_to_6d(root_quat_local.reshape(-1, 4)).reshape(n, t, 6)
  joint_pos_exp = (
    joint_pos
    if joint_pos.dim() >= 3
    else joint_pos.unsqueeze(1).expand(n, t, joint_pos.shape[-1])
  )
  joint_vel_exp = (
    joint_vel
    if joint_vel.dim() >= 3
    else joint_vel.unsqueeze(1).expand(n, t, joint_vel.shape[-1])
  )
  pos_obs_parts = [*root_pos_terms]
  if include_root_rot:
    pos_obs_parts.append(root_rot_6d)
  pos_obs_parts.append(joint_pos_exp)
  if extra_body_pos_w is not None:
    if extra_body_quat_w is None:
      raise ValueError(
        "extra_body_pos_w requires extra_body_quat_w."
      )
    n_b = extra_body_pos_w.shape[2]
    flat_n = n * t * n_b
    ap = root_pos[:, :, None, :].expand(n, t, n_b, 3).reshape(flat_n, 3)
    aq = root_quat[:, :, None, :].expand(n, t, n_b, 4).reshape(flat_n, 4)
    bp = extra_body_pos_w.reshape(flat_n, 3)
    bq = extra_body_quat_w.reshape(flat_n, 4)
    pos_b, _ = subtract_frame_transforms(ap, aq, bp, bq)
    pos_obs_parts.append(pos_b.reshape(n, t, n_b * 3))
  pos_obs = torch.cat(pos_obs_parts, dim=-1)
  vel_obs_parts: list[torch.Tensor] = []
  if include_projected_gravity:
    gravity_w = torch.zeros((n, t, 3), device=root_pos.device, dtype=root_pos.dtype)
    gravity_w[..., 2] = -1.0
    projected_gravity = quat_apply_inverse(
      root_quat_local.reshape(-1, 4), gravity_w.reshape(-1, 3)
    ).reshape(n, t, 3)
    vel_obs_parts.append(projected_gravity)
  if include_root_vel:
    vel_obs_parts.extend([root_lin_vel, root_ang_vel])
  vel_obs_parts.append(joint_vel_exp)
  vel_obs = torch.cat(vel_obs_parts, dim=-1)
  disc_obs = torch.cat([pos_obs, vel_obs], dim=-1).reshape(n, -1)
  return disc_obs


def calc_disc_obs_dim(
  num_disc_obs_steps: int,
  num_joints: int,
  root_height_obs: bool = True,
  include_root_xy: bool = True,
  include_root_rot: bool = True,
  include_root_vel: bool = True,
  include_projected_gravity: bool = False,
  num_disc_body_pos_b: int = 0,
) -> int:
  """Discriminator observation dimension."""
  pos_dim = num_joints
  if include_root_xy:
    pos_dim += 2
  if root_height_obs:
    pos_dim += 1
  if include_root_rot:
    pos_dim += 6
  pos_dim += 3 * num_disc_body_pos_b
  vel_dim = num_joints + (6 if include_root_vel else 0) + (3 if include_projected_gravity else 0)
  return num_disc_obs_steps * (pos_dim + vel_dim)


@dataclass
class AMPCfg:
  """Configuration for AMP in an env."""

  num_disc_obs_steps: int = 2
  asset_name: str = "robot"
  root_body_name: str = "LINK_BASE"
  """Body used as the AMP root for pos/quat/vel semantics."""
  motion_file: str | list[str] | None = None
  """Path to one csv or list of csv paths for reference motions."""
  global_obs: bool = False
  root_height_obs: bool = True
  include_root_xy: bool = True
  """Whether to include root relative x/y in discriminator observation."""
  include_root_rot: bool = True
  """Whether to include root 6D orientation in discriminator observation."""
  include_root_vel: bool = True
  """Whether to include root linear/angular velocity in discriminator observation."""
  include_projected_gravity: bool = False
  """Whether to include projected gravity in discriminator observation."""
  disc_body_pos_b_link_names: tuple[str, ...] = ()
  """Extra link positions in anchor frame, appended to disc obs."""


class AMPHelper:
  """Maintains state history and computes disc_obs for an RL env. Used when cfg.amp is set."""

  def __init__(self, env: ManagerBasedRlEnv, cfg: AMPCfg) -> None:
    self._env = env
    self._cfg = cfg
    self._device = env.device
    self._num_envs = env.num_envs
    robot = env.scene[cfg.asset_name]
    self._robot = robot
    self._root_body_idx = robot.body_names.index(cfg.root_body_name)
    self._extra_body_idx = tuple(
      robot.body_names.index(n) for n in cfg.disc_body_pos_b_link_names
    )
    self._num_disc_body_pos_b = len(self._extra_body_idx)
    self._num_joints = robot.data.joint_pos.shape[1]
    self._default_joint_pos = robot.data.default_joint_pos.clone()
    self._disc_dim = calc_disc_obs_dim(
      cfg.num_disc_obs_steps,
      self._num_joints,
      cfg.root_height_obs,
      cfg.include_root_xy,
      cfg.include_root_rot,
      cfg.include_root_vel,
      cfg.include_projected_gravity,
      self._num_disc_body_pos_b,
    )
    n = cfg.num_disc_obs_steps
    self._hist_root_pos = CircularBuffer(n, self._num_envs, self._device)
    self._hist_root_quat = CircularBuffer(n, self._num_envs, self._device)
    self._hist_root_lin = CircularBuffer(n, self._num_envs, self._device)
    self._hist_root_ang = CircularBuffer(n, self._num_envs, self._device)
    self._hist_joint_pos = CircularBuffer(n, self._num_envs, self._device)
    self._hist_joint_vel = CircularBuffer(n, self._num_envs, self._device)
    self._hist_extra_body_pos = CircularBuffer(n, self._num_envs, self._device)
    self._hist_extra_body_quat = CircularBuffer(n, self._num_envs, self._device)
    self._disc_obs_buf = torch.zeros(
      (self._num_envs, self._disc_dim), device=self._device, dtype=torch.float32
    )
    self._demo_data: list[dict[str, torch.Tensor]] | None = None
    self._demo_disc_obs: torch.Tensor | None = None
    self._demo_pair_states: torch.Tensor | None = None
    self._demo_pair_next_states: torch.Tensor | None = None
    if cfg.motion_file:
      paths = (
        cfg.motion_file
        if isinstance(cfg.motion_file, (list, tuple))
        else [cfg.motion_file]
      )
      self._load_demos(paths)

  def _load_demos(self, paths: list[str]) -> None:
    """Load demos from one or more motion files (npz). Each file becomes one motion."""
    self._demo_data = []
    for path in paths:
      self._demo_data.append(self._load_one_demo(path))
    self._build_demo_cache()

  def _load_one_demo(self, path: str) -> dict[str, torch.Tensor]:
    """Load a single motion demo from .npz."""
    if path.endswith(".npz"):
      # NPZ path: same semantics as tracking MotionLoader (joint_pos, joint_vel,
      # and either root_* or body_* arrays for determining root state).
      data = np.load(path)
      joint_pos = torch.from_numpy(data["joint_pos"]).float().to(self._device)
      joint_vel = torch.from_numpy(data["joint_vel"]).float().to(self._device)
      if joint_pos.ndim == 2:
        joint_pos = joint_pos.unsqueeze(0)
        joint_vel = joint_vel.unsqueeze(0)
      T = joint_pos.shape[1]

      if "root_pos" in data:
        root_pos = torch.from_numpy(data["root_pos"]).float().to(self._device)
        root_quat = torch.from_numpy(data["root_quat"]).float().to(self._device)
        root_lin = torch.from_numpy(data["root_lin_vel"]).float().to(self._device)
        root_ang = torch.from_numpy(data["root_ang_vel"]).float().to(self._device)
        if root_pos.ndim == 2:
          root_pos = root_pos.unsqueeze(0)
          root_quat = root_quat.unsqueeze(0)
          root_lin = root_lin.unsqueeze(0)
          root_ang = root_ang.unsqueeze(0)
      elif "body_pos_w" in data:
        body_pos = np.asarray(data["body_pos_w"])
        body_quat = np.asarray(data["body_quat_w"])
        if body_pos.ndim == 2:
          body_pos = body_pos[:, np.newaxis, :]
          body_quat = body_quat[:, np.newaxis, :]
        root_idx = self._root_body_idx
        root_pos = (
          torch.from_numpy(body_pos[:, root_idx, :]).float().to(self._device).unsqueeze(0)
        )
        root_quat = (
          torch.from_numpy(body_quat[:, root_idx, :]).float().to(self._device).unsqueeze(0)
        )
        if "body_lin_vel_w" in data and "body_ang_vel_w" in data:
          body_lin = np.asarray(data["body_lin_vel_w"])
          body_ang = np.asarray(data["body_ang_vel_w"])
          if body_lin.ndim == 2:
            body_lin = body_lin[:, np.newaxis, :]
            body_ang = body_ang[:, np.newaxis, :]
          root_lin = (
            torch.from_numpy(body_lin[:, root_idx, :]).float().to(self._device).unsqueeze(0)
          )
          root_ang = (
            torch.from_numpy(body_ang[:, root_idx, :]).float().to(self._device).unsqueeze(0)
          )
        else:
          root_lin = torch.zeros(1, T, 3, device=self._device)
          root_ang = torch.zeros(1, T, 3, device=self._device)
        out: dict[str, torch.Tensor] = {
          "root_pos": root_pos,
          "root_quat": root_quat,
          "root_lin_vel": root_lin,
          "root_ang_vel": root_ang,
          "joint_pos": joint_pos,
          "joint_vel": joint_vel,
        }
        if self._cfg.disc_body_pos_b_link_names:
          body_count = body_pos.shape[1]
          if self._root_body_idx >= body_count:
            raise ValueError(
              f"Motion npz '{path}' body count ({body_count}) is smaller than "
              f"root body index {self._root_body_idx}."
            )
          for bi in self._extra_body_idx:
            if bi >= body_count:
              raise ValueError(
                f"Motion npz '{path}' body count ({body_count}) is smaller than "
                f"required disc body index {bi}."
              )
          idx_np = np.asarray(self._extra_body_idx, dtype=np.int64)
          out["extra_body_pos_w"] = (
            torch.from_numpy(body_pos[:, idx_np, :]).float().to(self._device).unsqueeze(0)
          )
          out["extra_body_quat_w"] = (
            torch.from_numpy(body_quat[:, idx_np, :]).float().to(self._device).unsqueeze(0)
          )
        return out
      else:
        root_pos = torch.zeros(1, T, 3, device=self._device)
        root_quat = torch.zeros(1, T, 4, device=self._device)
        root_quat[:, :, 0] = 1.0
        root_lin = torch.zeros(1, T, 3, device=self._device)
        root_ang = torch.zeros(1, T, 3, device=self._device)

      out = {
        "root_pos": root_pos,
        "root_quat": root_quat,
        "root_lin_vel": root_lin,
        "root_ang_vel": root_ang,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
      }
      if self._cfg.disc_body_pos_b_link_names:
        raise ValueError(
          f"Motion npz '{path}' must provide body_pos_w/body_quat_w when "
          "disc_body_pos_b_link_names is configured."
        )
      return out
    raise ValueError(
      f"Unsupported AMP motion file format '{path}'. Expected a .npz file."
    )

  def _pad_demo_motion(
    self, demo: dict[str, torch.Tensor], target_len: int
  ) -> dict[str, torch.Tensor]:
    """Pad a demo to target_len by repeating the last frame."""
    cur_len = demo["root_pos"].shape[1]
    if cur_len >= target_len:
      return demo
    pad = target_len - cur_len
    padded: dict[str, torch.Tensor] = {}
    for key, value in demo.items():
      padded[key] = torch.cat(
        [value, value[:, -1:].expand(-1, pad, value.shape[-1])], dim=1
      )
    return padded

  def _compute_demo_disc_sequence(
    self, demo: dict[str, torch.Tensor]
  ) -> torch.Tensor | None:
    """Precompute disc_obs for every valid history window in one demo."""
    n_steps = self._cfg.num_disc_obs_steps
    demo = self._pad_demo_motion(demo, n_steps)
    seq_len = demo["root_pos"].shape[1]
    num_windows = seq_len - n_steps + 1
    if num_windows <= 0:
      return None

    def _windows(x: torch.Tensor) -> torch.Tensor:
      return torch.stack([x[0, i : i + n_steps] for i in range(num_windows)], dim=0)

    root_pos = _windows(demo["root_pos"])
    root_quat = _windows(demo["root_quat"])
    root_lin = _windows(demo["root_lin_vel"])
    root_ang = _windows(demo["root_ang_vel"])
    default_joint_pos = self._default_joint_pos[0:1]
    joint_pos = _windows(demo["joint_pos"] - default_joint_pos.unsqueeze(1))
    joint_vel = _windows(demo["joint_vel"])
    extra_kw: dict[str, torch.Tensor | None] = {
      "extra_body_pos_w": None,
      "extra_body_quat_w": None,
    }
    if self._cfg.disc_body_pos_b_link_names:
      extra_kw["extra_body_pos_w"] = _windows(demo["extra_body_pos_w"])
      extra_kw["extra_body_quat_w"] = _windows(demo["extra_body_quat_w"])
    return compute_disc_obs(
      ref_root_pos=root_pos[:, -1],
      ref_root_quat=root_quat[:, -1],
      root_pos=root_pos,
      root_quat=root_quat,
      root_lin_vel=root_lin,
      root_ang_vel=root_ang,
      joint_pos=joint_pos,
      joint_vel=joint_vel,
      global_obs=self._cfg.global_obs,
      root_height_obs=self._cfg.root_height_obs,
      include_root_xy=self._cfg.include_root_xy,
      include_root_rot=self._cfg.include_root_rot,
      include_root_vel=self._cfg.include_root_vel,
      include_projected_gravity=self._cfg.include_projected_gravity,
      **extra_kw,
    )

  def _build_demo_cache(self) -> None:
    """Precompute demo discriminator observations once at load time."""
    if not self._demo_data:
      self._demo_disc_obs = None
      self._demo_pair_states = None
      self._demo_pair_next_states = None
      return

    disc_sequences: list[torch.Tensor] = []
    pair_states: list[torch.Tensor] = []
    pair_next_states: list[torch.Tensor] = []
    for demo in self._demo_data:
      disc_seq = self._compute_demo_disc_sequence(demo)
      if disc_seq is None or disc_seq.numel() == 0:
        continue
      disc_sequences.append(disc_seq)
      if disc_seq.shape[0] > 1:
        pair_states.append(disc_seq[:-1])
        pair_next_states.append(disc_seq[1:])

    self._demo_disc_obs = torch.cat(disc_sequences, dim=0) if disc_sequences else None
    self._demo_pair_states = torch.cat(pair_states, dim=0) if pair_states else None
    self._demo_pair_next_states = (
      torch.cat(pair_next_states, dim=0) if pair_next_states else None
    )

  def update(self, env_ids: torch.Tensor | None = None) -> None:
    """Append current robot state to history and update disc_obs buffer."""
    r = self._robot.data
    root_pos = r.body_link_pos_w[:, self._root_body_idx]
    root_quat = r.body_link_quat_w[:, self._root_body_idx]
    root_lin = r.body_link_lin_vel_w[:, self._root_body_idx]
    root_ang = r.body_link_ang_vel_w[:, self._root_body_idx]
    jpos = r.joint_pos - self._default_joint_pos
    jvel = r.joint_vel
    self._hist_root_pos.append(root_pos)
    self._hist_root_quat.append(root_quat)
    self._hist_root_lin.append(root_lin)
    self._hist_root_ang.append(root_ang)
    self._hist_joint_pos.append(jpos)
    self._hist_joint_vel.append(jvel)
    if self._cfg.disc_body_pos_b_link_names:
      idx = torch.tensor(self._extra_body_idx, device=self._device, dtype=torch.long)
      extra_pos = r.body_link_pos_w.index_select(1, idx)
      extra_quat = r.body_link_quat_w.index_select(1, idx)
      self._hist_extra_body_pos.append(extra_pos)
      self._hist_extra_body_quat.append(extra_quat)
    if not self._hist_root_pos.is_initialized:
      return
    buf = self._hist_root_pos.buffer
    n, t = buf.shape[0], buf.shape[1]
    if t < self._cfg.num_disc_obs_steps:
      return
    ref_pos = self._hist_root_pos.buffer[:, -1]
    ref_quat = self._hist_root_quat.buffer[:, -1]
    extra_kw: dict[str, torch.Tensor | None] = {
      "extra_body_pos_w": None,
      "extra_body_quat_w": None,
    }
    if self._cfg.disc_body_pos_b_link_names:
      extra_kw["extra_body_pos_w"] = self._hist_extra_body_pos.buffer
      extra_kw["extra_body_quat_w"] = self._hist_extra_body_quat.buffer
    self._disc_obs_buf[:] = compute_disc_obs(
      ref_root_pos=ref_pos,
      ref_root_quat=ref_quat,
      root_pos=self._hist_root_pos.buffer,
      root_quat=self._hist_root_quat.buffer,
      root_lin_vel=self._hist_root_lin.buffer,
      root_ang_vel=self._hist_root_ang.buffer,
      joint_pos=self._hist_joint_pos.buffer,
      joint_vel=self._hist_joint_vel.buffer,
      global_obs=self._cfg.global_obs,
      root_height_obs=self._cfg.root_height_obs,
      include_root_xy=self._cfg.include_root_xy,
      include_root_rot=self._cfg.include_root_rot,
      include_root_vel=self._cfg.include_root_vel,
      include_projected_gravity=self._cfg.include_projected_gravity,
      **extra_kw,
    )

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Reset history for given envs (or all)."""
    if env_ids is None:
      self._hist_root_pos.reset(None)
      self._hist_root_quat.reset(None)
      self._hist_root_lin.reset(None)
      self._hist_root_ang.reset(None)
      self._hist_joint_pos.reset(None)
      self._hist_joint_vel.reset(None)
      self._hist_extra_body_pos.reset(None)
      self._hist_extra_body_quat.reset(None)
    else:
      self._hist_root_pos.reset(env_ids)
      self._hist_root_quat.reset(env_ids)
      self._hist_root_lin.reset(env_ids)
      self._hist_root_ang.reset(env_ids)
      self._hist_joint_pos.reset(env_ids)
      self._hist_joint_vel.reset(env_ids)
      self._hist_extra_body_pos.reset(env_ids)
      self._hist_extra_body_quat.reset(env_ids)

  def get_disc_obs(self) -> torch.Tensor:
    """Current disc_obs. Shape (num_envs, disc_dim)."""
    return self._disc_obs_buf

  def get_disc_obs_space(self) -> Box:
    """Gym-style Box space for disc_obs."""
    return Box(
      shape=(self._disc_dim,),
      low=-float("inf"),
      high=float("inf"),
      dtype="float32",
    )

  def fetch_disc_obs_demo(self, num_samples: int) -> torch.Tensor:
    """Sample num_samples demo disc_obs for discriminator training. Shape (num_samples, disc_dim)."""
    if self._demo_disc_obs is not None and self._demo_disc_obs.shape[0] > 0:
      indices = torch.randint(
        0, self._demo_disc_obs.shape[0], (num_samples,), device=self._device
      )
      return self._demo_disc_obs[indices]

    n_steps = self._cfg.num_disc_obs_steps
    # No motion file: synthetic standing (default pose, zero vel), use env 0 as ref
    default_pos = self._robot.data.default_joint_pos[0:1]  # (1, J)
    root_pos = self._robot.data.body_link_pos_w[0:1, self._root_body_idx].unsqueeze(1).expand(
      1, n_steps, 3
    )
    root_quat = self._robot.data.body_link_quat_w[0:1, self._root_body_idx].unsqueeze(1).expand(
      1, n_steps, 4
    )
    root_lin = torch.zeros(1, n_steps, 3, device=self._device)
    root_ang = torch.zeros(1, n_steps, 3, device=self._device)
    joint_pos = torch.zeros(1, n_steps, self._num_joints, device=self._device)
    joint_vel = torch.zeros(1, n_steps, self._num_joints, device=self._device)
    ref_pos = root_pos[:, -1]
    ref_quat = root_quat[:, -1]
    extra_kw: dict[str, torch.Tensor | None] = {
      "extra_body_pos_w": None,
      "extra_body_quat_w": None,
    }
    k = self._num_disc_body_pos_b
    if k:
      r0 = self._robot.data
      idx = torch.tensor(self._extra_body_idx, device=self._device, dtype=torch.long)
      extra_kw["extra_body_pos_w"] = r0.body_link_pos_w[0:1].index_select(
        1, idx
      ).unsqueeze(1).expand(1, n_steps, k, 3)
      extra_kw["extra_body_quat_w"] = r0.body_link_quat_w[0:1].index_select(
        1, idx
      ).unsqueeze(1).expand(1, n_steps, k, 4)
    one = compute_disc_obs(
      ref_root_pos=ref_pos,
      ref_root_quat=ref_quat,
      root_pos=root_pos,
      root_quat=root_quat,
      root_lin_vel=root_lin,
      root_ang_vel=root_ang,
      joint_pos=joint_pos,
      joint_vel=joint_vel,
      global_obs=self._cfg.global_obs,
      root_height_obs=self._cfg.root_height_obs,
      include_root_xy=self._cfg.include_root_xy,
      include_root_rot=self._cfg.include_root_rot,
      include_root_vel=self._cfg.include_root_vel,
      include_projected_gravity=self._cfg.include_projected_gravity,
      **extra_kw,
    )
    return one.expand(num_samples, -1)

  def fetch_disc_obs_demo_pairs(self, num_pairs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample num_pairs consecutive (s_t, s_{t+1}) from cached demo pairs."""
    if (
      self._demo_pair_states is not None
      and self._demo_pair_next_states is not None
      and self._demo_pair_states.shape[0] > 0
    ):
      indices = torch.randint(
        0, self._demo_pair_states.shape[0], (num_pairs,), device=self._device
      )
      return self._demo_pair_states[indices], self._demo_pair_next_states[indices]

    if self._demo_data is None or len(self._demo_data) == 0:
      single = self.fetch_disc_obs_demo(1)
      return single.expand(num_pairs, -1), single.expand(num_pairs, -1)
    single = self.fetch_disc_obs_demo(1)
    return single.expand(num_pairs, -1), single.expand(num_pairs, -1)
