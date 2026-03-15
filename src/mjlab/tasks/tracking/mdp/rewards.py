from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_error_magnitude,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Cached protector map: (y_grid, z_grid, values_2d, dy, dz) per path; dy,dz 由格心间距算一次供仿真复用
_protector_map_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]] = {}
_force_params_cache: dict[str, dict] = {}


def _load_yz_protector_tsv(
  path: Path,
  device: torch.device,
  dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
  """Load YZ protector TSV; 格心间距 dy, dz 在加载时算一次，供整个仿真查表复用。

  Returns:
    y_grid, z_grid, values_2d, dy, dz.
    values_2d[z_idx, y_idx] = thickness mm. dy, dz 为格心间距 (m)，由 (grid_max - grid_min)/(n-1) 得到。
  """
  path_str = str(path.resolve())
  if path_str in _protector_map_cache:
    return _protector_map_cache[path_str]

  lines = path.read_text().strip().splitlines()
  start = 0
  for i, line in enumerate(lines):
    if not line.strip().startswith("#"):
      start = i
      break
  header = lines[start].split("\t")
  y_grid = torch.tensor([float(x) for x in header[1:]], device=device, dtype=dtype)
  z_list = []
  values_list = []
  for line in lines[start + 1 :]:
    parts = line.split("\t")
    if not parts:
      continue
    z_list.append(float(parts[0]))
    values_list.append([float(parts[j]) for j in range(1, len(parts))])
  z_grid = torch.tensor(z_list, device=device, dtype=dtype)
  values_2d = torch.tensor(values_list, device=device, dtype=dtype)  # [nz, ny]
  ny, nz = y_grid.numel(), z_grid.numel()
  dy = float((y_grid[-1] - y_grid[0]) / max(ny - 1, 1))
  dz = float((z_grid[-1] - z_grid[0]) / max(nz - 1, 1))
  _protector_map_cache[path_str] = (y_grid, z_grid, values_2d, dy, dz)
  return y_grid, z_grid, values_2d, dy, dz


def _lookup_thickness_yz(
  y: torch.Tensor,
  z: torch.Tensor,
  y_grid: torch.Tensor,
  z_grid: torch.Tensor,
  values_2d: torch.Tensor,
  dy: float,
  dz: float,
) -> torch.Tensor:
  """Nearest-neighbor lookup: (y, z) -> thickness (mm). dy, dz 为格心间距（由 _load_yz_protector_tsv 算一次复用）。"""
  y_ = torch.clamp(y, y_grid[0].item(), y_grid[-1].item())
  z_ = torch.clamp(z, z_grid[0].item(), z_grid[-1].item())
  dy_safe = dy if dy > 0 else 1e-9
  dz_safe = dz if dz > 0 else 1e-9
  y_idx = ((y_ - y_grid[0]) / dy_safe).long().clamp(0, y_grid.numel() - 1)
  z_idx = ((z_ - z_grid[0]) / dz_safe).long().clamp(0, z_grid.numel() - 1)
  return values_2d[z_idx, y_idx]


def _load_force_params(path: Path) -> dict:
  """Load C, alpha, beta, gamma from fitted_parameters.json."""
  path_str = str(path.resolve())
  if path_str in _force_params_cache:
    return _force_params_cache[path_str]
  with open(path, "r", encoding="utf-8") as f:
    params = json.load(f)
  _force_params_cache[path_str] = params
  return params


def _force_after_protector_torch(
  F_before_kN: torch.Tensor,
  t_mm: torch.Tensor,
  p: float,
  C: float,
  alpha: float,
  beta: float,
  gamma: float,
) -> torch.Tensor:
  """护具衰减公式：F_after = C * (t^alpha) * (p^beta) * (F_before^gamma)。

  输入单位（与 scripts/ThicknessCalculate/force_calculator.py 一致）：
    t_mm: 厚度，单位 mm
    F_before_kN: 缓冲前冲击力，单位 kN
    p: 密度，无量纲
  输出：F_after 单位 kN。
  """
  # Where t_mm <= 0 or F_before_kN <= 0, return F_before_kN (no attenuation)
  t_safe = torch.clamp(t_mm, min=1e-6)
  F_safe = torch.clamp(F_before_kN, min=1e-9)
  F_after = C * (t_safe**alpha) * (p**beta) * (F_safe**gamma)
  return torch.where(
    (t_mm > 0) & (F_before_kN > 0),
    F_after,
    F_before_kN,
  )


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def _get_joint_indexes(
  asset: Entity, joint_patterns: tuple[str, ...]
) -> list[int]:
  """根据模式查找关节索引。

  Args:
    asset: Entity 对象
    joint_patterns: 关节名称模式元组，支持部分匹配（使用 in 操作符）

  Returns:
    匹配的关节索引列表
  """
  # 获取关节名称
  joint_names = getattr(asset, "joint_names", None)
  if joint_names is None:
    return []

  # 查找匹配的关节索引
  indices = []
  for pattern in joint_patterns:
    for i, name in enumerate(joint_names):
      if pattern in name:
        indices.append(i)
        break
  return indices


def _get_sensor_body_names(sensor: ContactSensor) -> list[str]:
  """Extract unique body names from sensor slots, preserving order."""
  body_names = []
  seen = set()
  for slot in sensor._slots:
    if slot.primary_name not in seen:
      body_names.append(slot.primary_name)
      seen.add(slot.primary_name)
  return body_names


def contact_pos_in_default_standing_frame(
  contact_pos_w: torch.Tensor,
  current_body_pos_w: torch.Tensor,
  current_body_quat_w: torch.Tensor,
  default_body_pos: torch.Tensor,
  default_body_quat: torch.Tensor,
) -> torch.Tensor:
  """把世界系下的接触点变换到「默认站立」世界系（与护具 map 同系）。

  从 base 到接触点经过整条运动链（一系列关节角），这里不手写链式变换，而是：
  1) 当前：current_body_pos_w / current_body_quat_w 来自 asset.data.body_link_pose_w，
     即仿真器已用当前 qpos 做完 FK，每个 body 的世界位姿已包含从 base 到该 body 的全部关节角。
  2) 接触点在该 body 局部系下的位置：p_body_local = current_body_quat^{-1} * (contact_pos_w - current_body_pos_w)，
     即「接触点相对该 body 原点的向量」在 body 系下的表示，与关节链无关。
  3) 默认：default_body_pos / default_body_quat 来自「整机 qpos 设为 default 再 forward」后的 body_link_pose_w，
     即仿真器用默认关节角做完 FK，同样已包含从 base 到该 body 的整条链。
  4) 用默认位姿把同一 body 局部点变回世界：p_default = default_body_pos + default_body_quat * p_body_local。

  因此关节链在「当前位姿」和「默认位姿」里都由仿真器 FK 正确体现，我们只做 body 局部 ↔ 世界 的转换。
  """
  delta_w = contact_pos_w - current_body_pos_w
  p_body_local = quat_apply_inverse(current_body_quat_w, delta_w)
  B = contact_pos_w.shape[0]
  default_pos_exp = default_body_pos.unsqueeze(0).expand(B, -1, -1)
  default_quat_exp = default_body_quat.unsqueeze(0).expand(B, -1, -1)
  p_default = default_pos_exp + quat_apply(default_quat_exp, p_body_local)
  return p_default


def get_default_body_poses_for_contact_sensor(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> tuple[torch.Tensor, torch.Tensor]:
  """在默认 qpos（XML 站立姿态，含根位姿 + 所有关节角）下算 contact sensor 对应 body 的世界位姿，并缓存。

  临时把整机 qpos 设为 default（root + 全部关节），forward 一次后读 body_link_pose_w，
  因此得到的 default 位姿已包含「从 base 经整条运动链到各 body」的 FK，再由 sim 恢复原 qpos。
  结果缓存在 env._contact_default_body_poses 上，key=(sensor_name, asset_name)。

  用法示例（在 reward 里做护具力衰减时）:
    default_pos, default_quat = get_default_body_poses_for_contact_sensor(env, sensor_name, asset_cfg)
    body_indices, _ = asset.find_bodies(_get_sensor_body_names(sensor), preserve_order=True)
    pose_w = asset.data.body_link_pose_w[:, body_indices, :]  # [B, N, 7]
    pos_b = contact_pos_in_default_standing_frame(
      sensor.data.pos, pose_w[..., :3], pose_w[..., 3:7], default_pos, default_quat
    )
    再用 pos_b 查护具 map 得到衰减系数即可。

  Returns:
    default_body_pos: [N, 3], default_body_quat: [N, 4]，N 为 sensor 的 primary body 数
  """
  cache = getattr(env, "_contact_default_body_poses", None)
  if cache is None:
    cache = {}
    setattr(env, "_contact_default_body_poses", cache)
  key = (sensor_name, asset_cfg.name)
  if key in cache:
    return cache[key]

  sensor: ContactSensor = env.scene[sensor_name]
  asset: Entity = env.scene[asset_cfg.name]
  body_names = _get_sensor_body_names(sensor)
  body_indices, matched_names = asset.find_bodies(body_names, preserve_order=True)
  if len(body_indices) != len(body_names):
    raise ValueError(
      f"Contact sensor has {len(body_names)} bodies but only {len(body_indices)} matched in asset '{asset_cfg.name}'. "
      f"Missing: {set(body_names) - set(matched_names)}"
    )

  qpos_save = env.sim.data.qpos.clone()
  root_pose = asset.data.default_root_state[:, :7]
  asset.data.write_root_pose(root_pose, env_ids=None)
  asset.data.write_joint_position(asset.data.default_joint_pos, env_ids=None)
  env.scene.write_data_to_sim()
  env.sim.forward()
  pose_w = asset.data.body_link_pose_w[:, body_indices, :]
  default_body_pos = pose_w[0, :, :3].clone()
  default_body_quat = pose_w[0, :, 3:7].clone()

  env.sim.data.qpos.copy_(qpos_save)
  env.scene.write_data_to_sim()
  env.sim.forward()

  cache[key] = (default_body_pos, default_body_quat)
  return default_body_pos, default_body_quat


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Cost that returns the number of self-collisions detected by a sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)


def reduce_contact_force_weighted(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  high_weight_bodies: tuple[str, ...] = (),
  medium_weight_bodies: tuple[str, ...] = (),
  high_weight: float = 10.0,
  medium_weight: float = 1.0,
  low_weight: float = 0.1,
  alpha: float = 0.3,
  protector_map_dir: Optional[Path] = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  force_params_path: Optional[Path] = None,
  density: float = 0.3,
) -> torch.Tensor:
  """Reward for reducing contact force based on paper formula.
  
  Implements: r_contact = (1/N) * Σ ||I{c_i} w_s,i f_contact,i||_2 + α * max ||I{c_i} w_s,i f_contact,i||_2
  If protector_map_dir is set, contact force is first attenuated by protector map (YZ TSV thickness
  lookup + force formula from scripts/ThicknessCalculate/force_calculator.py), then the weighted
  penalty is applied to the attenuated force.

  Units (计算全程单位约定):
    - Sensor force (sensor.data.force): N.
    - 没加 map 时（protector_map_dir is None）：直接用 sensor 力（N）算加权 penalty，不做衰减。
    - 加 map 时：TSV 厚度 mm → 公式得 F_after_kN → F_after_N = F_after_kN*1000；衰减后力向量 = 原方向 × F_after_N（单位 N）。
    - Reward: penalty = weighted force magnitude → 力单位 N。
  
  Args:
    env: The environment.
    sensor_name: Name of the contact sensor (e.g., "body_contact_force").
    high_weight_bodies: List of body names with high vulnerability (e.g., head, hands).
    medium_weight_bodies: List of body names with medium vulnerability (e.g., shanks, shoulders).
    high_weight: Sensitivity weight for high vulnerability bodies (default: 1000.0).
    medium_weight: Sensitivity weight for medium vulnerability bodies (default: 1.0).
    low_weight: Sensitivity weight for low vulnerability bodies (default: 0.5).
    alpha: Weight balancing average and peak forces (default: 0.3).
    protector_map_dir: If set, use YZ front/back TSV maps and force formula to attenuate force before penalty.
    asset_cfg: Asset config for default-standing pose (used when protector_map_dir is set).
    force_params_path: Path to fitted_parameters.json (default: protector_map_dir / "fitted_parameters.json").
    density: Material density for force formula (used when protector_map_dir is set).
    
  Returns:
    Reward tensor of shape [B] where higher values indicate lower contact forces.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  assert sensor.data.found is not None

  forces = sensor.data.force  # [B, N, 3]，单位 N
  contact_indicators = (sensor.data.found > 0).float()  # [B, N]
  device = forces.device
  dtype = forces.dtype

  # reward 只用到力的范数再按权重求和，故统一成 force_magnitude [B, N]（单位 N）
  force_magnitude = torch.norm(forces, dim=-1)
  if protector_map_dir is not None:
    assert sensor.data.pos is not None
    protector_map_dir = Path(protector_map_dir)
    default_pos, default_quat = get_default_body_poses_for_contact_sensor(
      env, sensor_name, asset_cfg
    )
    asset = env.scene[asset_cfg.name]
    body_names = _get_sensor_body_names(sensor)
    body_indices, _ = asset.find_bodies(body_names, preserve_order=True)
    pose_w = asset.data.body_link_pose_w[:, body_indices, :]  # [B, N, 7]
    pos_default = contact_pos_in_default_standing_frame(
      sensor.data.pos,
      pose_w[..., :3],
      pose_w[..., 3:7],
      default_pos,
      default_quat,
    )
    path_front = protector_map_dir / "yz_map_front.tsv"
    path_back = protector_map_dir / "yz_map_back.tsv"
    y_f, z_f, v_f, dy_f, dz_f = _load_yz_protector_tsv(path_front, device, dtype)
    y_b, z_b, v_b, dy_b, dz_b = _load_yz_protector_tsv(path_back, device, dtype)
    fp_path = (
      Path(force_params_path)
      if force_params_path is not None
      else protector_map_dir / "fitted_parameters.json"
    )
    params = _load_force_params(fp_path)
    C, alpha_f, beta, gamma = params["C"], params["alpha"], params["beta"], params["gamma"]
    x_d, y_d, z_d = pos_default[..., 0], pos_default[..., 1], pos_default[..., 2]
    t_front = _lookup_thickness_yz(y_d, z_d, y_f, z_f, v_f, dy_f, dz_f)
    t_back = _lookup_thickness_yz(y_d, z_d, y_b, z_b, v_b, dy_b, dz_b)
    t_mm = torch.where(x_d >= 0, t_front, t_back)
    F_before_kN = force_magnitude.clamp(min=1e-6) / 1000.0
    F_after_kN = _force_after_protector_torch(
      F_before_kN, t_mm, density, C, alpha_f, beta, gamma
    )
    force_magnitude = F_after_kN * 1000.0  # F_after_N，单位 N

  body_names = _get_sensor_body_names(sensor)
  weights = torch.full(
    (len(body_names),), low_weight, device=device, dtype=dtype
  )
  for i, body_name in enumerate(body_names):
    if body_name in high_weight_bodies:
      weights[i] = high_weight
    elif body_name in medium_weight_bodies:
      weights[i] = medium_weight

  # 加权力范数 = indicator * weight * |force|（只用到大小，单位 N）
  weighted_force_norm = contact_indicators * weights.unsqueeze(0) * force_magnitude
  num_active_contacts = contact_indicators.sum(dim=-1, keepdim=True).clamp(min=1.0)
  average_term = weighted_force_norm.sum(dim=-1) / num_active_contacts.squeeze(-1)
  peak_term = weighted_force_norm.max(dim=-1)[0]
  penalty = average_term + alpha * peak_term
  return -penalty


def feet_relative_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Reward tracking feet position relative to anchor/torso.

  Compares reference feet positions relative to anchor vs robot feet positions relative to anchor.
  Returns exp(-mean_error / std^2). If feet links not found, returns zeros.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  names = command.cfg.body_names
  device = (
    command.device
    if hasattr(command, "device")
    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
  )

  # Find left and right foot indices
  try:
    li = names.index("LINK_ANKLE_ROLL_L")
    ri = names.index("LINK_ANKLE_ROLL_R")
  except ValueError:
    return torch.zeros(env.num_envs, device=device)

  # Reference: feet positions relative to anchor (world frame)
  p_ref_l = command.body_pos_w[:, li] - command.anchor_pos_w  # (N, 3)
  p_ref_r = command.body_pos_w[:, ri] - command.anchor_pos_w  # (N, 3)

  # Robot: feet positions relative to anchor (world frame)
  p_rob_l = command.robot_body_pos_w[:, li] - command.robot_anchor_pos_w  # (N, 3)
  p_rob_r = command.robot_body_pos_w[:, ri] - command.robot_anchor_pos_w  # (N, 3)

  # Calculate squared errors for both feet
  err_l = torch.sum((p_ref_l - p_rob_l) ** 2, dim=-1)  # (N,)
  err_r = torch.sum((p_ref_r - p_rob_r) ** 2, dim=-1)  # (N,)

  # Average error over both feet
  mean_err = (err_l + err_r) / 2.0

  return torch.exp(-mean_err / (std**2 + 1e-9))


def projected_gravity_tracking_reward(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Reward for tracking projected gravity vector.

  Compares the gravity vector projected in the anchor frame between reference and robot.
  This helps the robot maintain the correct torso/anchor orientation.

  Returns exp(-||g_ref_b - g_robot_b||^2 / std^2) where:
    - g_ref_b: gravity projected in reference anchor frame
    - g_robot_b: gravity projected in robot anchor frame

  Args:
    env: Environment instance
    command_name: Name of the motion command
    std: Standard deviation for the exponential reward

  Returns:
    Reward tensor of shape (N,)
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset = env.scene[command.cfg.asset_name]

  # Get gravity vector in world frame (N, 3)
  g_w = asset.data.gravity_vec_w

  # Project gravity into reference anchor frame
  from mjlab.utils.lab_api.math import quat_apply_inverse

  g_ref_b = quat_apply_inverse(command.anchor_quat_w, g_w)

  # Project gravity into robot anchor frame
  g_robot_b = quat_apply_inverse(command.robot_anchor_quat_w, g_w)

  # Calculate squared error
  error = torch.sum((g_ref_b - g_robot_b) ** 2, dim=-1)  # (N,)

  return torch.exp(-error / (std**2 + 1e-9))


def ankle_pitch_joint_tracking_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """脚踝 pitch 关节跟踪奖励：比较当前与参考的脚踝 pitch 关节角，返回 exp(-avg_err/std^2)。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset: Entity = env.scene[asset_cfg.name]
  # 查找左右脚踝 pitch 关节索引
  pitch_indices = _get_joint_indexes(
    asset, ("J04_ANKLE_PITCH_L", "J10_ANKLE_PITCH_R")
  )
  if not pitch_indices:
    device = getattr(command, "device", None) or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.zeros(env.num_envs, device=device)
  cur = asset.data.joint_pos
  ref = command.joint_pos
  errs = [(cur[:, idx] - ref[:, idx]) ** 2 for idx in pitch_indices]
  avg_err = torch.stack(errs, dim=0).mean(dim=0)
  return torch.exp(-avg_err / (std**2 + 1e-9))


def ankle_roll_joint_tracking_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """脚踝 roll 关节跟踪奖励：比较当前与参考的脚踝 roll 关节角，返回 exp(-avg_err/std^2)。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset: Entity = env.scene[asset_cfg.name]
  # 查找左右脚踝 roll 关节索引
  roll_indices = _get_joint_indexes(
    asset, ("J05_ANKLE_ROLL_L", "J11_ANKLE_ROLL_R")
  )
  if not roll_indices:
    device = getattr(command, "device", None) or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.zeros(env.num_envs, device=device)
  cur = asset.data.joint_pos
  ref = command.joint_pos
  errs = [(cur[:, idx] - ref[:, idx]) ** 2 for idx in roll_indices]
  avg_err = torch.stack(errs, dim=0).mean(dim=0)
  return torch.exp(-avg_err / (std**2 + 1e-9))


# def foot_slip_penalty(
#   env: ManagerBasedRlEnv,
#   command_name: str,
#   asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
#   contact_threshold: float = 1.0,
#   foot_contact_sensor_names: Optional[list[str]] = None,
# ) -> torch.Tensor:
#   """Foot slip penalty based on Isaac Gym implementation.

#   Penalizes foot velocity when the foot is in contact with the ground.
#   Uses contact sensors (similar to self_collision_cost) for accurate contact detection.

#   Args:
#     env: The environment instance
#     command_name: Name of the motion command (for getting foot indices if needed)
#     asset_cfg: Asset configuration for the robot
#     contact_threshold: Minimum contact force threshold to consider foot in contact
#     foot_contact_sensor_names: List of contact sensor names for feet (e.g., ["left_foot_contact", "right_foot_contact"])

#   Returns:
#     Penalty value for each environment (higher values mean more slippage)
#   """
#   asset: Entity = env.scene[asset_cfg.name]
#   command = cast(MotionCommand, env.command_manager.get_term(command_name))

#   # Look for foot body indices - try common foot body names
#   body_names = getattr(command.cfg, "body_names", [])
#   foot_indices = []

#   # Common foot body names to search for
#   foot_body_patterns = [
#     # "LINK_FOOT_R", "LINK_FOOT_L",
#     "LINK_ANKLE_ROLL_L",
#     "LINK_ANKLE_ROLL_R",
#   ]

#   for i, body_name in enumerate(body_names):
#     for pattern in foot_body_patterns:
#       if pattern in body_name:
#         foot_indices.append(i)
#         break

#   # # Debug: Print foot indices found
#   # if hasattr(env, '_foot_indices_debug_printed') is False:
#   #   print(f"[FOOT_SLIP_DEBUG] Body names: {body_names}")
#   #   print(f"[FOOT_SLIP_DEBUG] Found foot indices: {foot_indices}")
#   #   if foot_indices:
#   #     foot_body_names = [body_names[i] for i in foot_indices]
#   #     print(f"[FOOT_SLIP_DEBUG] Foot body names: {foot_body_names}")
#   #   env._foot_indices_debug_printed = True

#   # If no foot indices found, return zeros
#   if not foot_indices:
#     device = getattr(command, "device", None) or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     return torch.zeros(env.num_envs, device=device)

#   foot_indices = torch.tensor(foot_indices, device=getattr(command, "device", None))

#   # Get foot velocities - shape: (num_envs, num_feet, 3)
#   foot_vel = command.robot_body_lin_vel_w[:, foot_indices]

#   # Get contact forces using contact sensors (similar to self_collision_cost API)
#   contact_forces_magnitude = []
#   sensors_used = []

#   if foot_contact_sensor_names:
#     # Use contact sensors for more accurate contact detection
#     for sensor_name in foot_contact_sensor_names:
#       try:
#         # Get contact sensor data - similar to self_collision_cost
#         sensor: ContactSensor = env.scene[sensor_name]
#         assert sensor.data.force is not None
#         # Get contact force magnitude for each foot
#         # sensor.data.force shape: [B, N, 3] where N is number of primary objects
#         contact_force_mag = torch.norm(sensor.data.force, dim=-1)  # [B, N]
#         # Take max over all contacts for this sensor (in case multiple contacts per foot)
#         contact_force_z = contact_force_mag.max(dim=-1)[0] if contact_force_mag.shape[-1] > 0 else torch.zeros(env.num_envs, device=foot_vel.device)

#         contact_forces_magnitude.append(contact_force_z)
#         sensors_used.append(sensor_name)
#       except KeyError:
#         # Sensor not found, skip it
#         continue

#   # If no contact sensors found or no contact forces, assume no contact
#   if not contact_forces_magnitude:
#     device = foot_vel.device
#     contact_force_magnitude = torch.zeros(
#       (env.num_envs, len(foot_indices)), device=device
#     )
#   else:
#     # Stack contact forces for all feet - shape: (num_envs, num_feet)
#     contact_force_magnitude = torch.stack(contact_forces_magnitude, dim=1)

#   # Calculate contact mask: True where contact force magnitude > threshold
#   in_contact = contact_force_magnitude > contact_threshold

#   # Calculate horizontal (XY) foot speed, as per the reference
#   foot_speed_xy = torch.norm(foot_vel[..., :2], dim=-1)  # (num_envs, num_feet)

#   # Penalty is the square root of horizontal speed, scaled by contact
#   rew = torch.sqrt(foot_speed_xy)
#   rew *= in_contact.float()
#   penalty = torch.sum(rew, dim=1)  # (num_envs,)

#   return penalty


def ankle_joint_smoothness_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """脚踝关节平滑惩罚：惩罚脚踝关节的急剧变化，防止抖动

  计算脚踝关节速度的变化率（加速度），对过大的加速度进行惩罚。
  返回负值惩罚：-acceleration^2 / std^2
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset: Entity = env.scene[asset_cfg.name]

  # 查找脚踝关节索引
  ankle_indices = _get_joint_indexes(
    asset, ("J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R")
  )
  if not ankle_indices:
    device = getattr(command, "device", None) or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.zeros(env.num_envs, device=device)

  # 获取当前关节速度
  current_joint_vel = asset.data.joint_vel[
    :, ankle_indices
  ]  # (num_envs, num_ankle_joints)

  # 初始化历史关节速度（如果尚未初始化）
  if env._prev_ankle_joint_vel is None:
    env._prev_ankle_joint_vel = current_joint_vel.clone()
    return torch.zeros(env.num_envs, device=current_joint_vel.device)

  # 对于重置的环境，用当前值重新初始化
  reset_mask = env.episode_length_buf == 0
  if reset_mask.any():
    env._prev_ankle_joint_vel[reset_mask] = current_joint_vel[reset_mask].clone()

  # 计算关节加速度（速度变化率）
  dt = env.physics_dt * 4  # 控制步长
  joint_acceleration = (current_joint_vel - env._prev_ankle_joint_vel) / dt

  # 计算加速度的L2范数
  acceleration_magnitude = torch.norm(joint_acceleration, dim=-1)  # (num_envs,)

  # 惩罚：加速度越大，惩罚越大
  smoothness_penalty = -acceleration_magnitude  # / (std**2 + 1e-9)

  # 更新历史速度
  env._prev_ankle_joint_vel = current_joint_vel.clone()

  return smoothness_penalty


# 重力加速度 m/s^2，用于 10g 阈值
_GRAVITY = 9.81


def torso_acceleration_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  scale: float = 1.0,
  linear_only: bool = False,
  threshold_g: float = 10.0,
  body_name: str = "LINK_TORSO_YAW",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """指定 body（默认 torso）的加速度惩罚：仅当该 body 加速度超过 threshold_g 倍重力时才惩罚。

  body_name 为 root（如 LINK_BASE）且非固定基座时，使用仿真器 root_link_acc_w；
  否则使用该 body 的 link 速度差分得到加速度（与 ankle 平滑惩罚类似）。
  linear_only=True 时只考虑线加速度，否则线+角加速度一起算范数。

  Returns:
    负值惩罚，仅在超过阈值时有非零惩罚。
  """
  asset: Entity = env.scene[asset_cfg.name]
  device = asset.data.body_link_vel_w.device
  body_indices, _ = asset.find_bodies((body_name,), preserve_order=True)
  if len(body_indices) == 0:
    return torch.zeros(env.num_envs, device=device)

  body_idx = int(body_indices[0])
  is_root = body_idx == 0

  if is_root and not asset.data.is_fixed_base:
    acc_w = asset.data.root_link_acc_w  # (num_envs, 6)
  else:
    # 非 root：用速度差分得到加速度
    vel_w = asset.data.body_link_vel_w[:, body_idx, :]  # (num_envs, 6)
    cache = getattr(env, "_prev_body_vel_acc", None)
    if cache is None:
      cache = {}
      setattr(env, "_prev_body_vel_acc", cache)
    key = body_name
    if key not in cache:
      cache[key] = vel_w.clone()
      return torch.zeros(env.num_envs, device=device)
    dt = env.step_dt
    acc_w = (vel_w - cache[key]) / dt
    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
      cache[key] = cache[key].clone()
      cache[key][reset_mask] = vel_w[reset_mask]
    else:
      cache[key] = vel_w.clone()

  if linear_only:
    acc_w = acc_w[:, 0:3]
  acc_mag = torch.norm(acc_w, dim=-1)  # (num_envs,)
  threshold = threshold_g * _GRAVITY  # m/s^2
  excess = torch.clamp(acc_mag - threshold, min=0.0)  # 仅超出部分
  return -scale * (excess**2)


def joint_wrench_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  scale: float = 1.0,
  threshold: float = 500.0,
  wrench_type: str = "total",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """关节广义力（joint wrench）惩罚：仅当关节总广义力范数超过阈值时才惩罚。

  使用 EntityData 的 joint_qfrc_*。wrench_type 可选 "total"（默认）、"constraint"、"actuator"。
  对每个 env 取关节广义力向量的 L2 范数，超过 threshold（单位与 qfrc 一致，如 Nm）时对超出部分做平方惩罚。

  Returns:
    负值惩罚，仅在超过阈值时有非零惩罚。
  """
  asset: Entity = env.scene[asset_cfg.name]
  if not asset.data.is_articulated:
    return torch.zeros(env.num_envs, device=asset.data.joint_pos.device)
  if wrench_type == "total":
    qfrc = asset.data.joint_qfrc_total  # (num_envs, num_joint_dofs)
  elif wrench_type == "constraint":
    qfrc = asset.data.joint_qfrc_constraint
  elif wrench_type == "actuator":
    qfrc = asset.data.joint_qfrc_actuator
  else:
    raise ValueError(f"joint_wrench_penalty: unknown wrench_type={wrench_type!r}")
  wrench_norm = torch.norm(qfrc, dim=-1)  # (num_envs,)
  excess = torch.clamp(wrench_norm - threshold, min=0.0)
  return -scale * (excess**2)


def ankle_joint_jerk_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """脚踝关节急动度惩罚：惩罚加速度的变化率（jerk），进一步平滑运动

  Jerk是加速度的导数，过大的jerk会导致运动不平滑。
  返回负值惩罚：-jerk^2 / std^2
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset: Entity = env.scene[asset_cfg.name]

  # 查找脚踝关节索引
  ankle_indices = _get_joint_indexes(
    asset, ("J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R")
  )
  if not ankle_indices:
    device = getattr(command, "device", None) or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.zeros(env.num_envs, device=device)

  # 获取当前关节速度
  current_joint_vel = asset.data.joint_vel[:, ankle_indices]

  # 初始化历史数据（如果尚未初始化）
  if env._prev_ankle_joint_vel_jerk is None or env._prev_ankle_joint_acc is None:
    env._prev_ankle_joint_vel_jerk = current_joint_vel.clone()
    env._prev_ankle_joint_acc = torch.zeros_like(current_joint_vel)
    return torch.zeros(env.num_envs, device=current_joint_vel.device)

  # 对于重置的环境，用当前值重新初始化
  reset_mask = env.episode_length_buf == 0
  if reset_mask.any():
    env._prev_ankle_joint_vel_jerk[reset_mask] = current_joint_vel[reset_mask].clone()
    env._prev_ankle_joint_acc[reset_mask] = 0.0

  # 计算加速度
  dt = env.physics_dt * 4
  current_acceleration = (current_joint_vel - env._prev_ankle_joint_vel_jerk) / dt

  # 计算急动度（加速度的变化率）
  jerk = (current_acceleration - env._prev_ankle_joint_acc) / dt

  # 计算急动度的L2范数
  jerk_magnitude = torch.norm(jerk, dim=-1)

  # 惩罚：急动度越大，惩罚越大
  jerk_penalty = -jerk_magnitude  # / (std**2 + 1e-9)

  # 更新历史数据
  env._prev_ankle_joint_vel_jerk = current_joint_vel.clone()
  env._prev_ankle_joint_acc = current_acceleration.clone()

  return jerk_penalty


def ankle_joint_power_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """脚踝关节能量消耗惩罚：惩罚高功率消耗

  基于 power = |velocity| * |torque| 计算脚踝关节的功率消耗
  返回负值惩罚：-sum(|dof_vel| * |torques|) for ankle joints
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset: Entity = env.scene[asset_cfg.name]

  # 查找脚踝关节索引
  ankle_indices = _get_joint_indexes(
    asset, ("J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R")
  )
  if not ankle_indices:
    device = getattr(command, "device", None) or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.zeros(env.num_envs, device=device)

  # 获取脚踝关节速度和力矩
  ankle_joint_vel = asset.data.joint_vel[
    :, ankle_indices
  ]  # (num_envs, num_ankle_joints)
  ankle_joint_torques = asset.data.actuator_force[
    :, ankle_indices
  ]  # (num_envs, num_ankle_joints)

  # 计算功率：|velocity| * |torque|
  ankle_power = torch.abs(ankle_joint_vel) * torch.abs(ankle_joint_torques)

  # 对所有脚踝关节功率求和
  total_ankle_power = torch.sum(ankle_power, dim=-1)  # (num_envs,)

  # 返回负值惩罚
  return -total_ankle_power


def reward_feet_distance(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """
  脚部距离惩罚奖励函数：防止左右脚过于接近，使用渐进式惩罚策略

  设计原理：
  1. 使用全局坐标系计算距离（更稳定，不受机器人姿态影响）
  2. 多区间惩罚：危险区（强惩罚）+ 警告区（轻惩罚）+ 安全区（无惩罚）
  3. 只在脚部过近时惩罚，避免与其他奖励冲突
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  # 获取左右脚踝的body索引
  body_names = getattr(command.cfg, "body_names", [])

  try:
    left_ankle_idx = body_names.index("LINK_ANKLE_ROLL_L")
    right_ankle_idx = body_names.index("LINK_ANKLE_ROLL_R")
  except ValueError:
    # 如果找不到对应的body名称，返回零奖励
    device = getattr(
      command, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    return torch.zeros(env.num_envs, device=device)

  # 获取左右脚踝在全局坐标系下的位置
  left_ankle_pos = command.robot_body_pos_w[:, left_ankle_idx, :3].clone()
  right_ankle_pos = command.robot_body_pos_w[:, right_ankle_idx, :3].clone()

  # 计算脚踝间的横向距离（主要关注Y轴方向，即左右距离）
  lateral_diff = left_ankle_pos[:, 1] - right_ankle_pos[:, 1]  # Y轴差值
  lateral_distance = torch.abs(lateral_diff)  # 绝对距离

  # 也考虑X-Z平面的距离，防止前后交叉
  xz_diff = torch.stack(
    [
      left_ankle_pos[:, 0] - right_ankle_pos[:, 0],
      left_ankle_pos[:, 2] - right_ankle_pos[:, 2],
    ],
    dim=1,
  )
  xz_distance = torch.norm(xz_diff, dim=1)

  # 综合距离（主要是横向，辅以前后高度）
  total_distance = torch.sqrt(lateral_distance**2 + 0.5 * xz_distance**2)

  # 定义距离阈值（基于机器人髋部宽度）
  critical_distance = 0.06  # 6cm - 危险区（脚部几乎碰撞）
  warning_distance = 0.09  # 9cm - 警告区（过于接近）
  safe_distance = 0.12  # 12cm - 安全区（正常距离）

  # 渐进式惩罚策略
  # 危险区：强指数惩罚
  critical_penalty = torch.where(
    total_distance < critical_distance,
    5.0 * torch.exp(-10.0 * total_distance),  # 强惩罚，快速增长
    torch.zeros_like(total_distance),
  )

  # 警告区：线性惩罚
  warning_penalty = torch.where(
    (total_distance >= critical_distance) & (total_distance < warning_distance),
    2.0 * (warning_distance - total_distance) / (warning_distance - critical_distance),
    torch.zeros_like(total_distance),
  )

  # 轻微警告区：很小的惩罚，平滑过渡
  mild_penalty = torch.where(
    (total_distance >= warning_distance) & (total_distance < safe_distance),
    0.5 * (safe_distance - total_distance) / (safe_distance - warning_distance),
    torch.zeros_like(total_distance),
  )

  # 总惩罚（负奖励）
  total_penalty = critical_penalty + warning_penalty + mild_penalty

  return -total_penalty