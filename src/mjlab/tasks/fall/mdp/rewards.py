from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_error_magnitude,
  subtract_frame_transforms,
)
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def _get_sensor_body_names(sensor: ContactSensor) -> list[str]:
  """Extract unique body names from sensor slots, preserving order."""
  body_names = []
  seen = set()
  for slot in sensor._slots:
    if slot.primary_name not in seen:
      body_names.append(slot.primary_name)
      seen.add(slot.primary_name)
  return body_names


def _build_body_weight_tensor(
  body_names: list[str],
  device: torch.device,
  dtype: torch.dtype,
  high_weight_bodies: tuple[str, ...] = (),
  shoulder_weight_bodies: tuple[str, ...] = (),
  medium_weight_bodies: tuple[str, ...] = (),
  high_weight: float = 10.0,
  shoulder_weight: float = 5.0,
  medium_weight: float = 1.0,
  low_weight: float = 0.1,
) -> torch.Tensor:
  """Build per-body vulnerability weights matching contact-force reward groups."""
  weights = torch.full((len(body_names),), low_weight, device=device, dtype=dtype)
  for i, body_name in enumerate(body_names):
    if body_name in high_weight_bodies:
      weights[i] = high_weight
    elif body_name in shoulder_weight_bodies:
      weights[i] = shoulder_weight
    elif body_name in medium_weight_bodies:
      weights[i] = medium_weight
  return weights

def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize self-collisions.

  Returns the number of self-collisions detected by the specified contact sensor.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)


def control_descent_speed(
  env: ManagerBasedRlEnv,
  torso_body_name: str = "LINK_TORSO_YAW",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  threshold: float = 0.5,
) -> torch.Tensor:
  """Penalize torso downward speed beyond a safe threshold."""
  asset: Entity = env.scene[asset_cfg.name]
  body_ids, _ = asset.find_bodies((torso_body_name,), preserve_order=True)
  if not body_ids:
    return torch.zeros(env.num_envs, device=env.device)
  torso_lin_vel_z = asset.data.body_link_lin_vel_w[:, body_ids[0], 2]
  downward_speed = torch.clamp(-torso_lin_vel_z - threshold, min=0.0)
  return -(downward_speed**2)


class ImpactVelocityReward:
  """Penalize each body's first ground contact once per episode."""

  def __init__(
    self,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    high_weight_bodies: tuple[str, ...] = (),
    shoulder_weight_bodies: tuple[str, ...] = (),
    medium_weight_bodies: tuple[str, ...] = (),
    high_weight: float = 10.0,
    shoulder_weight: float = 5.0,
    medium_weight: float = 1.0,
    low_weight: float = 0.1,
  ) -> None:
    self.sensor_name = sensor_name
    self.asset_cfg = asset_cfg
    self.high_weight_bodies = high_weight_bodies
    self.shoulder_weight_bodies = shoulder_weight_bodies
    self.medium_weight_bodies = medium_weight_bodies
    self.high_weight = high_weight
    self.shoulder_weight = shoulder_weight
    self.medium_weight = medium_weight
    self.low_weight = low_weight
    self._body_ids: list[int] | None = None
    self._body_names: list[str] | None = None
    self._contacted_once: torch.Tensor | None = None

  def _maybe_initialize(self, env: ManagerBasedRlEnv) -> bool:
    sensor: ContactSensor = env.scene[self.sensor_name]
    assert sensor.data.found is not None
    if self._body_names is None:
      self._body_names = _get_sensor_body_names(sensor)
    if not self._body_names:
      return False
    if self._body_ids is None:
      asset: Entity = env.scene[self.asset_cfg.name]
      self._body_ids, _ = asset.find_bodies(
        tuple(self._body_names), preserve_order=True
      )
    if not self._body_ids:
      return False
    if self._contacted_once is None or self._contacted_once.shape != sensor.data.found.shape:
      self._contacted_once = torch.zeros_like(sensor.data.found, dtype=torch.bool)
    return True

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._contacted_once is None:
      return
    if env_ids is None:
      self._contacted_once[:] = False
    else:
      self._contacted_once[env_ids] = False

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    if not self._maybe_initialize(env):
      return torch.zeros(env.num_envs, device=env.device)

    sensor: ContactSensor = env.scene[self.sensor_name]
    assert sensor.data.found is not None
    assert self._body_names is not None
    assert self._body_ids is not None
    assert self._contacted_once is not None

    contact_now = sensor.data.found > 0
    first_contact_once = contact_now & ~self._contacted_once

    asset: Entity = env.scene[self.asset_cfg.name]
    body_lin_vel = asset.data.body_link_lin_vel_w[:, self._body_ids, :]
    downward_speed = torch.clamp(-body_lin_vel[..., 2], min=0.0)
    weights = _build_body_weight_tensor(
      body_names=self._body_names,
      device=body_lin_vel.device,
      dtype=body_lin_vel.dtype,
      high_weight_bodies=self.high_weight_bodies,
      shoulder_weight_bodies=self.shoulder_weight_bodies,
      medium_weight_bodies=self.medium_weight_bodies,
      high_weight=self.high_weight,
      shoulder_weight=self.shoulder_weight,
      medium_weight=self.medium_weight,
      low_weight=self.low_weight,
    )
    weighted_impact = (
      first_contact_once.float() * downward_speed.square() * weights.unsqueeze(0)
    )
    num_impacts = torch.clamp(first_contact_once.float().sum(dim=-1), min=1.0)
    penalty = weighted_impact.sum(dim=-1) / num_impacts
    self._contacted_once |= contact_now
    return -penalty

def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost

def reduce_contact_force_weighted(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  high_weight_bodies: tuple[str, ...] = (),
  shoulder_weight_bodies: tuple[str, ...] = (),
  medium_weight_bodies: tuple[str, ...] = (),
  high_weight: float = 10.0,
  shoulder_weight: float = 5.0,
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
    shoulder_weight_bodies: List of shoulder body names with dedicated vulnerability weight.
    medium_weight_bodies: List of body names with medium vulnerability.
    high_weight: Sensitivity weight for high vulnerability bodies (default: 1000.0).
    shoulder_weight: Sensitivity weight for shoulder bodies (default: 5.0).
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
  weights = _build_body_weight_tensor(
    body_names=body_names,
    device=forces.device,
    dtype=forces.dtype,
    high_weight_bodies=high_weight_bodies,
    shoulder_weight_bodies=shoulder_weight_bodies,
    medium_weight_bodies=medium_weight_bodies,
    high_weight=high_weight,
    shoulder_weight=shoulder_weight,
    medium_weight=medium_weight,
    low_weight=low_weight,
  )
  
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


def ang_vel_xy(
  env: ManagerBasedRlEnv,
  target_base_height_phase3: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward low root angular velocity around x/y after stand-up height is reached."""
  asset: Entity = env.scene[asset_cfg.name]
  standup = (asset.data.root_link_pos_w[:, 2] > target_base_height_phase3).float()
  reward = torch.exp(torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1) * -2.0)
  return reward * standup


def lin_vel_xy(
  env: ManagerBasedRlEnv,
  target_base_height_phase3: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward low root linear velocity around x/y after stand-up height is reached."""
  asset: Entity = env.scene[asset_cfg.name]
  standup = (asset.data.root_link_pos_w[:, 2] > target_base_height_phase3).float()
  reward = torch.exp(torch.sum(torch.square(asset.data.root_link_lin_vel_b[:, :2]), dim=1) * -5.0)
  return reward * standup


def target_orientation(
  env: ManagerBasedRlEnv,
  target_base_height_phase3: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward upright orientation after stand-up height is reached."""
  asset: Entity = env.scene[asset_cfg.name]
  standup = (asset.data.root_link_pos_w[:, 2] > target_base_height_phase3).float()
  reward = torch.exp(torch.sum(torch.square(asset.data.projected_gravity_b[:, :1]), dim=1) * -5.0)
  return reward * standup


def target_base_height(
  env: ManagerBasedRlEnv,
  base_height_target: float,
  target_base_height_phase3: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward root height near target after stand-up height is reached."""
  asset: Entity = env.scene[asset_cfg.name]
  base_height = asset.data.root_link_pos_w[:, 2]
  standup = (base_height > target_base_height_phase3).float()
  reward = torch.exp(torch.abs(base_height - base_height_target) * -20.0)
  return reward * standup


def initial_pose_error_exp(
  env: ManagerBasedRlEnv,
  target_base_height_phase3: float,
  std: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward keeping XML/MJCF default joint pose after stand-up.

  Target pose comes from ``asset.data.default_joint_pos`` loaded from robot XML/MJCF.
  """
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  if default_joint_pos is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  standup = (asset.data.root_link_pos_w[:, 2] > target_base_height_phase3).float()
  joint_error = torch.sum(torch.square(asset.data.joint_pos - default_joint_pos), dim=1)
  reward = torch.exp(-joint_error / (std**2))
  return reward * standup


def _resolve_body_ids_for_initial_pose_reward(
  asset: Entity,
  body_names: tuple[str, ...] | None,
) -> torch.Tensor:
  if body_names:
    body_ids, _ = asset.find_bodies(body_names, preserve_order=True)
    if not body_ids:
      raise ValueError(
        "initial body pose reward got empty body_names; no matching bodies found."
      )
    return torch.tensor(body_ids, device=asset.data.body_link_pos_w.device, dtype=torch.long)

  # By default use all bodies except root link, so reward focuses on articulated posture.
  if asset.num_bodies <= 1:
    return torch.zeros((0,), device=asset.data.body_link_pos_w.device, dtype=torch.long)
  return torch.arange(1, asset.num_bodies, device=asset.data.body_link_pos_w.device, dtype=torch.long)


def _get_initial_body_pose_targets(
  env: ManagerBasedRlEnv,
  asset: Entity,
  body_names: tuple[str, ...] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
  env_any = cast(Any, env)
  cache = getattr(env_any, "_fall_initial_body_pose_targets_cache", None)
  if cache is None:
    cache = {}
    setattr(env_any, "_fall_initial_body_pose_targets_cache", cache)

  cache_key = (id(asset), tuple(body_names) if body_names else None)
  cached = cache.get(cache_key, None)
  if cached is not None:
    return cached["body_ids"], cached["target_pos_b"], cached["target_quat_b"]

  body_ids = _resolve_body_ids_for_initial_pose_reward(asset, body_names)
  if body_ids.numel() == 0:
    target_pos_b = torch.zeros((0, 3), device=env.device, dtype=torch.float32)
    target_quat_b = torch.zeros((0, 4), device=env.device, dtype=torch.float32)
    target_quat_b[:, 0] = 1.0
    cache[cache_key] = {
      "body_ids": body_ids,
      "target_pos_b": target_pos_b,
      "target_quat_b": target_quat_b,
    }
    return body_ids, target_pos_b, target_quat_b

  # Keep reward computation side-effect free: never write simulator state here.
  # Lock target only when at least one env is clearly upright.
  standup_candidates = torch.nonzero(asset.data.root_link_pos_w[:, 2] > 0.65, as_tuple=False).squeeze(-1)
  if standup_candidates.numel() == 0:
    return None
  candidate_heights = asset.data.root_link_pos_w[standup_candidates, 2]
  ref_env_idx = int(standup_candidates[torch.argmax(candidate_heights)].item())
  ref_env_ids = torch.tensor([ref_env_idx], device=env.device, dtype=torch.long)
  root_pos_w = asset.data.root_link_pos_w[ref_env_ids].expand(body_ids.numel(), 3)
  root_quat_w = asset.data.root_link_quat_w[ref_env_ids].expand(body_ids.numel(), 4)
  body_pos_w = asset.data.body_link_pos_w[ref_env_ids].index_select(1, body_ids).reshape(-1, 3)
  body_quat_w = asset.data.body_link_quat_w[ref_env_ids].index_select(1, body_ids).reshape(-1, 4)
  target_pos_b, target_quat_b = subtract_frame_transforms(
    root_pos_w, root_quat_w, body_pos_w, body_quat_w
  )

  cache[cache_key] = {
    "body_ids": body_ids,
    "target_pos_b": target_pos_b,
    "target_quat_b": target_quat_b,
  }
  return body_ids, target_pos_b, target_quat_b


def initial_body_position_error_exp(
  env: ManagerBasedRlEnv,
  target_base_height_phase3: float,
  std: float = 0.3,
  body_names: tuple[str, ...] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward body position (in root frame) close to XML/MJCF default pose after stand-up."""
  asset: Entity = env.scene[asset_cfg.name]
  standup = (asset.data.root_link_pos_w[:, 2] > target_base_height_phase3).float()
  if torch.max(standup).item() <= 0.0:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  target = _get_initial_body_pose_targets(env, asset, body_names)
  if target is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  body_ids, target_pos_b, _ = target
  if body_ids.numel() == 0:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  n = env.num_envs
  b = body_ids.numel()
  root_pos_w = asset.data.root_link_pos_w[:, None, :].expand(n, b, 3).reshape(-1, 3)
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(n, b, 4).reshape(-1, 4)
  body_pos_w = asset.data.body_link_pos_w.index_select(1, body_ids).reshape(-1, 3)
  body_quat_w = asset.data.body_link_quat_w.index_select(1, body_ids).reshape(-1, 4)
  body_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
  pos_error = torch.sum(
    torch.square(body_pos_b.reshape(n, b, 3) - target_pos_b[None, :, :]),
    dim=-1,
  )
  std_safe = max(float(std), 1e-6)
  reward = torch.exp(-pos_error.mean(-1) / (std_safe**2))
  return reward * standup


def initial_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  target_base_height_phase3: float,
  std: float = 0.4,
  body_names: tuple[str, ...] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward body orientation (in root frame) close to XML/MJCF default pose after stand-up."""
  asset: Entity = env.scene[asset_cfg.name]
  standup = (asset.data.root_link_pos_w[:, 2] > target_base_height_phase3).float()
  if torch.max(standup).item() <= 0.0:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  target = _get_initial_body_pose_targets(env, asset, body_names)
  if target is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  body_ids, _, target_quat_b = target
  if body_ids.numel() == 0:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  n = env.num_envs
  b = body_ids.numel()
  root_pos_w = asset.data.root_link_pos_w[:, None, :].expand(n, b, 3).reshape(-1, 3)
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(n, b, 4).reshape(-1, 4)
  body_pos_w = asset.data.body_link_pos_w.index_select(1, body_ids).reshape(-1, 3)
  body_quat_w = asset.data.body_link_quat_w.index_select(1, body_ids).reshape(-1, 4)
  _, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
  ori_error = quat_error_magnitude(
    body_quat_b.reshape(n, b, 4),
    target_quat_b[None, :, :].expand(n, b, 4),
  ) ** 2
  std_safe = max(float(std), 1e-6)
  reward = torch.exp(-ori_error.mean(-1) / (std_safe**2))
  return reward * standup

  