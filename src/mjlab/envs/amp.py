"""AMP (Adversarial Motion Priors) support for manager-based RL envs.

Provides disc_obs (discriminator observation) from robot state history,
get_disc_obs_space(), and fetch_disc_obs_demo() for use with AMP-style training
(e.g. MimicKit amp_agent, or amp-rsl-rl).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

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


def _extract_sorted_suffix_indices(fieldnames: Sequence[str], prefix: str) -> list[int]:
  indices: list[int] = []
  for name in fieldnames:
    if not name.startswith(prefix):
      continue
    suffix = name[len(prefix) :]
    if suffix.isdigit():
      indices.append(int(suffix))
  return sorted(indices)


def _csv_body_idx(asset_body_idx: int) -> int:
  """CSV body ids follow MuJoCo nbody and include world at index 0."""
  return asset_body_idx + 1


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
  include_root_lin_vel: bool = True,
  include_projected_gravity: bool = False,
  anchor_pos_w: torch.Tensor | None = None,
  anchor_quat_w: torch.Tensor | None = None,
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
  gravity_w = torch.tensor([0.0, 0.0, -1.0], device=root_quat.device, dtype=root_quat.dtype)
  gravity_w = gravity_w.reshape(1, 1, 3).expand(n, t, 3)
  projected_gravity = quat_apply_inverse(
    root_quat.reshape(-1, 4), gravity_w.reshape(-1, 3)
  ).reshape(n, t, 3)
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
  if include_projected_gravity:
    pos_obs_parts.append(projected_gravity)
  pos_obs_parts.append(joint_pos_exp)
  if extra_body_pos_w is not None:
    if (
      anchor_pos_w is None
      or anchor_quat_w is None
      or extra_body_quat_w is None
    ):
      raise ValueError(
        "extra_body_pos_w requires anchor_pos_w, anchor_quat_w, and extra_body_quat_w."
      )
    n_b = extra_body_pos_w.shape[2]
    flat_n = n * t * n_b
    ap = anchor_pos_w[:, :, None, :].expand(n, t, n_b, 3).reshape(flat_n, 3)
    aq = anchor_quat_w[:, :, None, :].expand(n, t, n_b, 4).reshape(flat_n, 4)
    bp = extra_body_pos_w.reshape(flat_n, 3)
    bq = extra_body_quat_w.reshape(flat_n, 4)
    pos_b, _ = subtract_frame_transforms(ap, aq, bp, bq)
    pos_obs_parts.append(pos_b.reshape(n, t, n_b * 3))
  pos_obs = torch.cat(pos_obs_parts, dim=-1)
  vel_obs_parts = []
  if include_root_lin_vel:
    vel_obs_parts.append(root_lin_vel)
  vel_obs_parts.extend([root_ang_vel, joint_vel_exp])
  vel_obs = torch.cat(vel_obs_parts, dim=-1)
  disc_obs = torch.cat([pos_obs, vel_obs], dim=-1).reshape(n, -1)
  return disc_obs


def calc_disc_obs_dim(
  num_disc_obs_steps: int,
  num_joints: int,
  root_height_obs: bool = True,
  include_root_xy: bool = True,
  include_root_rot: bool = True,
  include_root_lin_vel: bool = True,
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
  if include_projected_gravity:
    pos_dim += 3
  pos_dim += 3 * num_disc_body_pos_b
  vel_dim = (3 if include_root_lin_vel else 0) + 3 + num_joints
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
  include_root_lin_vel: bool = True
  """Whether to include root linear velocity in discriminator observation."""
  include_projected_gravity: bool = False
  """Whether to include projected gravity in root frame in discriminator observation."""
  disc_body_pos_b_link_names: tuple[str, ...] = ()
  """Extra link positions in the anchor frame (same transform as tracking `robot_body_pos_b`)."""
  disc_body_pos_b_anchor_body_name: str | None = None
  """Anchor body for `disc_body_pos_b_link_names`; None uses `root_body_name`."""


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
    anchor_name = cfg.disc_body_pos_b_anchor_body_name or cfg.root_body_name
    self._anchor_body_idx = robot.body_names.index(anchor_name)
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
      cfg.include_root_lin_vel,
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
    self._hist_anchor_pos = CircularBuffer(n, self._num_envs, self._device)
    self._hist_anchor_quat = CircularBuffer(n, self._num_envs, self._device)
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
    """Load demos from one or more csv/npz files. Each file becomes one motion."""
    self._demo_data = []
    for path in paths:
      self._demo_data.append(self._load_one_demo(path))
    self._build_demo_cache()

  def _load_one_demo(self, path: str) -> dict[str, torch.Tensor]:
    """Load a single motion demo from flattened csv or npz."""
    if path.endswith(".npz"):
      return self._load_one_demo_npz(path)
    if self._cfg.disc_body_pos_b_link_names:
      raise ValueError(
        "AMPCfg.disc_body_pos_b_link_names requires motion demos as .npz with "
        f"body_pos_w/body_quat_w; csv is not supported ({path})."
      )

    with open(path, newline="", encoding="utf-8") as csv_file:
      reader = csv.DictReader(csv_file)
      fieldnames = reader.fieldnames or []
      rows = list(reader)

    if not rows:
      raise ValueError(f"Motion csv '{path}' is empty.")

    joint_indices = _extract_sorted_suffix_indices(fieldnames, "joint_pos_")
    if not joint_indices or joint_indices != list(range(len(joint_indices))):
      raise ValueError(
        f"Motion csv '{path}' must provide contiguous joint_pos_i columns starting at 0."
      )
    joint_vel_indices = _extract_sorted_suffix_indices(fieldnames, "joint_vel_")
    if joint_vel_indices != joint_indices:
      raise ValueError(
        f"Motion csv '{path}' must provide matching joint_pos_i / joint_vel_i columns."
      )

    joint_pos_cols = [f"joint_pos_{joint_idx}" for joint_idx in joint_indices]
    joint_vel_cols = [f"joint_vel_{joint_idx}" for joint_idx in joint_indices]
    root_idx = _csv_body_idx(self._root_body_idx)
    root_pos_cols = [f"body_pos_w_{root_idx}_{axis}" for axis in ("x", "y", "z")]
    root_quat_cols = [f"body_quat_w_{root_idx}_{axis}" for axis in ("w", "x", "y", "z")]
    root_lin_cols = [f"body_lin_vel_w_{root_idx}_{axis}" for axis in ("x", "y", "z")]
    root_ang_cols = [f"body_ang_vel_w_{root_idx}_{axis}" for axis in ("x", "y", "z")]

    required_cols = joint_pos_cols + joint_vel_cols + root_pos_cols + root_quat_cols
    missing_cols = [col for col in required_cols if col not in fieldnames]
    if missing_cols:
      raise ValueError(
        f"Motion csv '{path}' is missing required columns: "
        f"{', '.join(missing_cols[:8])}"
        + ("..." if len(missing_cols) > 8 else "")
      )

    joint_pos = torch.tensor(
      [[float(row[col]) for col in joint_pos_cols] for row in rows],
      dtype=torch.float32,
      device=self._device,
    ).unsqueeze(0)
    joint_vel = torch.tensor(
      [[float(row[col]) for col in joint_vel_cols] for row in rows],
      dtype=torch.float32,
      device=self._device,
    ).unsqueeze(0)
    root_pos = torch.tensor(
      [[float(row[col]) for col in root_pos_cols] for row in rows],
      dtype=torch.float32,
      device=self._device,
    ).unsqueeze(0)
    root_quat = torch.tensor(
      [[float(row[col]) for col in root_quat_cols] for row in rows],
      dtype=torch.float32,
      device=self._device,
    ).unsqueeze(0)

    has_root_lin = all(col in fieldnames for col in root_lin_cols)
    has_root_ang = all(col in fieldnames for col in root_ang_cols)
    if has_root_lin != has_root_ang:
      raise ValueError(
        f"Motion csv '{path}' must contain both root linear/angular velocity "
        "columns or neither of them."
      )
    if has_root_lin:
      root_lin = torch.tensor(
        [[float(row[col]) for col in root_lin_cols] for row in rows],
        dtype=torch.float32,
        device=self._device,
      ).unsqueeze(0)
      root_ang = torch.tensor(
        [[float(row[col]) for col in root_ang_cols] for row in rows],
        dtype=torch.float32,
        device=self._device,
      ).unsqueeze(0)
    else:
      t = joint_pos.shape[1]
      root_lin = torch.zeros(1, t, 3, device=self._device)
      root_ang = torch.zeros(1, t, 3, device=self._device)

    return {
      "root_pos": root_pos,
      "root_quat": root_quat,
      "root_lin_vel": root_lin,
      "root_ang_vel": root_ang,
      "joint_pos": joint_pos,
      "joint_vel": joint_vel,
    }

  @staticmethod
  def _is_placeholder_root_sequence(
    pos: torch.Tensor, quat: torch.Tensor, lin: torch.Tensor, ang: torch.Tensor
  ) -> bool:
    """Detect world-like placeholder root sequence (all-zero pose/vel + identity quat)."""
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat.device, dtype=quat.dtype)
    pos_is_zero = torch.max(torch.abs(pos)).item() < 1e-6
    lin_is_zero = torch.max(torch.abs(lin)).item() < 1e-6
    ang_is_zero = torch.max(torch.abs(ang)).item() < 1e-6
    quat_is_identity = torch.max(torch.abs(quat - identity.unsqueeze(0))).item() < 1e-6
    return pos_is_zero and lin_is_zero and ang_is_zero and quat_is_identity

  def _load_one_demo_npz(self, path: str) -> dict[str, torch.Tensor]:
    """Load a single motion demo from npz."""
    npz = np.load(path, allow_pickle=True)
    required = (
      "joint_pos",
      "joint_vel",
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
    )
    missing = [key for key in required if key not in npz]
    if missing:
      raise ValueError(
        f"Motion npz '{path}' is missing required keys: {', '.join(missing)}"
      )

    joint_pos = torch.as_tensor(npz["joint_pos"], dtype=torch.float32, device=self._device)
    joint_vel = torch.as_tensor(npz["joint_vel"], dtype=torch.float32, device=self._device)
    body_pos_w = torch.as_tensor(npz["body_pos_w"], dtype=torch.float32, device=self._device)
    body_quat_w = torch.as_tensor(npz["body_quat_w"], dtype=torch.float32, device=self._device)
    body_lin_vel_w = torch.as_tensor(
      npz["body_lin_vel_w"], dtype=torch.float32, device=self._device
    )
    body_ang_vel_w = torch.as_tensor(
      npz["body_ang_vel_w"], dtype=torch.float32, device=self._device
    )

    if joint_pos.ndim != 2 or joint_vel.ndim != 2:
      raise ValueError(
        f"Motion npz '{path}' expects joint_pos/joint_vel shape (T, J), got "
        f"{tuple(joint_pos.shape)} and {tuple(joint_vel.shape)}."
      )
    if (
      body_pos_w.ndim != 3
      or body_quat_w.ndim != 3
      or body_lin_vel_w.ndim != 3
      or body_ang_vel_w.ndim != 3
    ):
      raise ValueError(
        f"Motion npz '{path}' expects body_*_w shape (T, B, D)."
      )

    root_idx = self._root_body_idx
    body_count = body_pos_w.shape[1]
    if root_idx >= body_count:
      raise ValueError(
        f"Motion npz '{path}' body count ({body_count}) is smaller than "
        f"required root index {root_idx}."
      )
    chosen_idx = root_idx
    if root_idx + 1 < body_count:
      root_seq = (
        body_pos_w[:, root_idx, :],
        body_quat_w[:, root_idx, :],
        body_lin_vel_w[:, root_idx, :],
        body_ang_vel_w[:, root_idx, :],
      )
      next_seq = (
        body_pos_w[:, root_idx + 1, :],
        body_quat_w[:, root_idx + 1, :],
        body_lin_vel_w[:, root_idx + 1, :],
        body_ang_vel_w[:, root_idx + 1, :],
      )
      if self._is_placeholder_root_sequence(*root_seq) and not self._is_placeholder_root_sequence(
        *next_seq
      ):
        chosen_idx = root_idx + 1

    root_pos = body_pos_w[:, chosen_idx, :].unsqueeze(0)
    root_quat = body_quat_w[:, chosen_idx, :].unsqueeze(0)
    root_lin = body_lin_vel_w[:, chosen_idx, :].unsqueeze(0)
    root_ang = body_ang_vel_w[:, chosen_idx, :].unsqueeze(0)
    out: dict[str, torch.Tensor] = {
      "root_pos": root_pos,
      "root_quat": root_quat,
      "root_lin_vel": root_lin,
      "root_ang_vel": root_ang,
      "joint_pos": joint_pos.unsqueeze(0),
      "joint_vel": joint_vel.unsqueeze(0),
    }
    if self._cfg.disc_body_pos_b_link_names:
      if self._anchor_body_idx >= body_count:
        raise ValueError(
          f"Motion npz '{path}' body count ({body_count}) is smaller than "
          f"anchor body index {self._anchor_body_idx}."
        )
      for bi in self._extra_body_idx:
        if bi >= body_count:
          raise ValueError(
            f"Motion npz '{path}' body count ({body_count}) is smaller than "
            f"required disc body index {bi}."
          )
      out["anchor_pos"] = body_pos_w[:, self._anchor_body_idx, :].unsqueeze(0)
      out["anchor_quat"] = body_quat_w[:, self._anchor_body_idx, :].unsqueeze(0)
      idx_t = torch.tensor(self._extra_body_idx, device=self._device, dtype=torch.long)
      out["extra_body_pos_w"] = body_pos_w[:, idx_t, :].unsqueeze(0)
      out["extra_body_quat_w"] = body_quat_w[:, idx_t, :].unsqueeze(0)
    return out

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
    anchor_kw: dict[str, torch.Tensor | None] = {
      "anchor_pos_w": None,
      "anchor_quat_w": None,
      "extra_body_pos_w": None,
      "extra_body_quat_w": None,
    }
    if self._cfg.disc_body_pos_b_link_names:
      anchor_kw["anchor_pos_w"] = _windows(demo["anchor_pos"])
      anchor_kw["anchor_quat_w"] = _windows(demo["anchor_quat"])
      anchor_kw["extra_body_pos_w"] = _windows(demo["extra_body_pos_w"])
      anchor_kw["extra_body_quat_w"] = _windows(demo["extra_body_quat_w"])
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
      include_root_lin_vel=self._cfg.include_root_lin_vel,
      include_projected_gravity=self._cfg.include_projected_gravity,
      **anchor_kw,
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
      anchor_pos = r.body_link_pos_w[:, self._anchor_body_idx]
      anchor_quat = r.body_link_quat_w[:, self._anchor_body_idx]
      idx = torch.tensor(self._extra_body_idx, device=self._device, dtype=torch.long)
      extra_pos = r.body_link_pos_w.index_select(1, idx)
      extra_quat = r.body_link_quat_w.index_select(1, idx)
      self._hist_anchor_pos.append(anchor_pos)
      self._hist_anchor_quat.append(anchor_quat)
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
    anchor_kw: dict[str, torch.Tensor | None] = {
      "anchor_pos_w": None,
      "anchor_quat_w": None,
      "extra_body_pos_w": None,
      "extra_body_quat_w": None,
    }
    if self._cfg.disc_body_pos_b_link_names:
      anchor_kw["anchor_pos_w"] = self._hist_anchor_pos.buffer
      anchor_kw["anchor_quat_w"] = self._hist_anchor_quat.buffer
      anchor_kw["extra_body_pos_w"] = self._hist_extra_body_pos.buffer
      anchor_kw["extra_body_quat_w"] = self._hist_extra_body_quat.buffer
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
      include_root_lin_vel=self._cfg.include_root_lin_vel,
      include_projected_gravity=self._cfg.include_projected_gravity,
      **anchor_kw,
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
      self._hist_anchor_pos.reset(None)
      self._hist_anchor_quat.reset(None)
      self._hist_extra_body_pos.reset(None)
      self._hist_extra_body_quat.reset(None)
    else:
      self._hist_root_pos.reset(env_ids)
      self._hist_root_quat.reset(env_ids)
      self._hist_root_lin.reset(env_ids)
      self._hist_root_ang.reset(env_ids)
      self._hist_joint_pos.reset(env_ids)
      self._hist_joint_vel.reset(env_ids)
      self._hist_anchor_pos.reset(env_ids)
      self._hist_anchor_quat.reset(env_ids)
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
    anchor_kw: dict[str, torch.Tensor | None] = {
      "anchor_pos_w": None,
      "anchor_quat_w": None,
      "extra_body_pos_w": None,
      "extra_body_quat_w": None,
    }
    k = self._num_disc_body_pos_b
    if k:
      r0 = self._robot.data
      anchor_kw["anchor_pos_w"] = r0.body_link_pos_w[
        0:1, self._anchor_body_idx
      ].unsqueeze(1).expand(1, n_steps, 3)
      anchor_kw["anchor_quat_w"] = r0.body_link_quat_w[
        0:1, self._anchor_body_idx
      ].unsqueeze(1).expand(1, n_steps, 4)
      idx = torch.tensor(self._extra_body_idx, device=self._device, dtype=torch.long)
      anchor_kw["extra_body_pos_w"] = r0.body_link_pos_w[0:1].index_select(
        1, idx
      ).unsqueeze(1).expand(1, n_steps, k, 3)
      anchor_kw["extra_body_quat_w"] = r0.body_link_quat_w[0:1].index_select(
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
      include_root_lin_vel=self._cfg.include_root_lin_vel,
      include_projected_gravity=self._cfg.include_projected_gravity,
      **anchor_kw,
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
