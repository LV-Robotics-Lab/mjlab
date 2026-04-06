from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _apply_mimic_phase_gate(
  env: "ManagerBasedRlEnv", value: torch.Tensor
) -> torch.Tensor:
  """Gate mimic/command-tracking reward terms during recovery.

  When `env.recovery_mode_buf` is True, we return 0 to disable mimic rewards.
  This keeps room for future recovery-specific reward terms that are not
  gated.
  """
  recovery_mode_buf = getattr(env, "recovery_mode_buf", None)
  if recovery_mode_buf is None:
    return value
  mask = (~recovery_mode_buf.to(device=value.device)).to(dtype=value.dtype)
  return value * mask


def _apply_mimic_phase_recovery_weight_scale(
  env: "ManagerBasedRlEnv",
  value: torch.Tensor,
  *,
  recovery_scale: float,
) -> torch.Tensor:
  """Mimic tracking at full scale; during recovery multiply by ``recovery_scale``."""
  recovery_mode_buf = getattr(env, "recovery_mode_buf", None)
  if recovery_mode_buf is None:
    return value
  r = recovery_mode_buf.to(device=value.device, dtype=value.dtype)
  scale = (1.0 - r) + float(recovery_scale) * r
  return value * scale


def _apply_recovery_phase_gate(
  env: "ManagerBasedRlEnv", value: torch.Tensor
) -> torch.Tensor:
  """Gate recovery-only rewards outside recovery mode."""
  recovery_mode_buf = getattr(env, "recovery_mode_buf", None)
  if recovery_mode_buf is None:
    return torch.zeros_like(value)
  mask = recovery_mode_buf.to(device=value.device, dtype=value.dtype)
  return value * mask


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


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  value = torch.exp(-error / std**2)
  return _apply_mimic_phase_recovery_weight_scale(env, value, recovery_scale=0.1)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  value = torch.exp(-error / std**2)
  return _apply_mimic_phase_recovery_weight_scale(env, value, recovery_scale=0.1)


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
  value = torch.exp(-error.mean(-1) / std**2)
  return _apply_mimic_phase_recovery_weight_scale(env, value, recovery_scale=0.1)


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
  value = torch.exp(-error.mean(-1) / std**2)
  return _apply_mimic_phase_recovery_weight_scale(env, value, recovery_scale=0.1)


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
  value = torch.exp(-error.mean(-1) / std**2)
  return _apply_mimic_phase_gate(env, value)


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
  value = torch.exp(-error.mean(-1) / std**2)
  return _apply_mimic_phase_gate(env, value)


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
) -> torch.Tensor:
  """Reward for reducing contact force based on paper formula.
  
  Implements: r_contact = (1/N) * Σ ||I{c_i} w_s,i f_contact,i||_2 + α * max ||I{c_i} w_s,i f_contact,i||_2
  
  Args:
    env: The environment.
    sensor_name: Name of the contact sensor (e.g., "body_contact_force").
    high_weight_bodies: List of body names with high vulnerability (e.g., head, hands).
    medium_weight_bodies: List of body names with medium vulnerability (e.g., shanks, shoulders).
    high_weight: Sensitivity weight for high vulnerability bodies (default: 1000.0).
    medium_weight: Sensitivity weight for medium vulnerability bodies (default: 1.0).
    low_weight: Sensitivity weight for low vulnerability bodies (default: 0.5).
    alpha: Weight balancing average and peak forces (default: 0.3).
    
  Returns:
    Reward tensor of shape [B] where higher values indicate lower contact forces.
    This is a penalty (negative reward), so higher values mean less penalty.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  assert sensor.data.found is not None
  
  # Get contact forces: [B, N, 3] where N is number of primary objects (bodies)
  forces = sensor.data.force  # [B, N, 3]
  
  # Get contact indicators: [B, N] (1 if in contact, 0 otherwise)
  contact_indicators = (sensor.data.found > 0).float()  # [B, N]
  
  # Get body names from sensor and create weight tensor
  body_names = _get_sensor_body_names(sensor)
  
  # Create sensitivity weight tensor: [N]
  weights = torch.full(
    (len(body_names),), low_weight, device=forces.device, dtype=forces.dtype
  )
  for i, body_name in enumerate(body_names):
    if body_name in high_weight_bodies:
      weights[i] = high_weight
    elif body_name in medium_weight_bodies:
      weights[i] = medium_weight
  
  # Compute weighted contact force magnitude: [B, N]
  # ||I{c_i} w_s,i f_contact,i||_2
  weighted_force_magnitude = (
    contact_indicators.unsqueeze(-1) * weights.unsqueeze(0).unsqueeze(-1) * forces
  )  # [B, N, 3]
  weighted_force_norm = torch.norm(weighted_force_magnitude, dim=-1)  # [B, N]
  
  # Count active contacts: N = Σ I{c_i}
  num_active_contacts = contact_indicators.sum(dim=-1, keepdim=True)  # [B, 1]
  # Avoid division by zero
  num_active_contacts = torch.clamp(num_active_contacts, min=1.0)
  
  # Average term: (1/N) * Σ ||I{c_i} w_s,i f_contact,i||_2
  average_term = weighted_force_norm.sum(dim=-1) / num_active_contacts.squeeze(-1)  # [B]
  
  # Peak term: max ||I{c_i} w_s,i f_contact,i||_2
  peak_term = weighted_force_norm.max(dim=-1)[0]  # [B]
  
  # Combined penalty: r_contact = average_term + α * peak_term
  penalty = average_term + alpha * peak_term  # [B]
  
  # Return negative penalty as reward (higher reward = less penalty)
  return -penalty


def recovery_reduce_contact_force_weighted(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  high_weight_bodies: tuple[str, ...] = (),
  medium_weight_bodies: tuple[str, ...] = (),
  high_weight: float = 10.0,
  medium_weight: float = 1.0,
  low_weight: float = 0.1,
  alpha: float = 0.3,
) -> torch.Tensor:
  """Recovery-only contact-force reward."""
  value = reduce_contact_force_weighted(
    env=env,
    sensor_name=sensor_name,
    high_weight_bodies=high_weight_bodies,
    medium_weight_bodies=medium_weight_bodies,
    high_weight=high_weight,
    medium_weight=medium_weight,
    low_weight=low_weight,
    alpha=alpha,
  )
  return _apply_recovery_phase_gate(env, value)


def recovery_body_height_penalty(
  env: ManagerBasedRlEnv,
  body_name: str = "LINK_HEAD_YAW",
  command_name: str = "motion",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  penalty_scale: float = 20.0,
) -> torch.Tensor:
  """Recovery-only: penalize chosen body below motion reference height (frozen frame)."""
  env_any = cast(Any, env)
  asset: Entity = env.scene[asset_cfg.name]
  if not hasattr(env_any, "_recovery_body_height_sim_idx"):
    env_any._recovery_body_height_sim_idx = asset.body_names.index(body_name)
  sim_body_idx = env_any._recovery_body_height_sim_idx
  sim_height = asset.data.body_link_pos_w[:, sim_body_idx, 2]

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if not hasattr(env_any, "_recovery_body_height_cmd_idx"):
    env_any._recovery_body_height_cmd_idx = command.cfg.body_names.index(body_name)
  cmd_body_idx = env_any._recovery_body_height_cmd_idx
  ref_height = command.body_pos_w[:, cmd_body_idx, 2]

  height_drop = torch.relu(ref_height - sim_height)
  value = -penalty_scale * height_drop
  return _apply_recovery_phase_gate(env, value)


def recovery_time_penalty(
  env: ManagerBasedRlEnv,
  per_step_penalty: float = 0.02,
) -> torch.Tensor:
  """Recovery-only time penalty to encourage faster exits."""
  value = torch.full(
    (env.num_envs,),
    -abs(float(per_step_penalty)),
    device=env.device,
    dtype=torch.float32,
  )
  return _apply_recovery_phase_gate(env, value)


def recovery_success_bonus(
  env: ManagerBasedRlEnv,
  bonus_scale: float = 3.0,
) -> torch.Tensor:
  """One-step scaled bonus emitted when recovery exits successfully."""
  bonus = getattr(env, "recovery_success_bonus_buf", None)
  if bonus is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  return bonus.to(device=env.device, dtype=torch.float32) * float(bonus_scale)


def recovery_entry_penalty_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """进入 recovery 当步的一次性负奖励（由 termination 写入 buffer）。

  Buffer 固定为 ``1.0``；总惩罚幅度只通过本 term 在 cfg 里的 ``weight`` 调节。
  RewardManager 会对返回值再乘 ``weight * dt``，因此这里除以 ``step_dt``，使
  ``weight=W`` 时该步贡献约为 ``-W``。
  """
  env_any = cast(Any, env)
  buf = getattr(env_any, "recovery_entry_penalty_buf", None)
  if buf is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  scale = float(getattr(env_any, "recovery_entry_penalty_scale", 1.0))
  out = -(buf * scale / max(float(env.step_dt), 1e-9))
  buf.zero_()
  return out.to(dtype=torch.float32)


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
  value = torch.exp(-mean_err / (std**2 + 1e-9))
  return _apply_mimic_phase_gate(env, value)


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
  value = torch.exp(-error / (std**2 + 1e-9))
  return _apply_mimic_phase_gate(env, value)


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
  value = torch.exp(-avg_err / (std**2 + 1e-9))
  return _apply_mimic_phase_gate(env, value)


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
  value = torch.exp(-avg_err / (std**2 + 1e-9))
  return _apply_mimic_phase_gate(env, value)


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