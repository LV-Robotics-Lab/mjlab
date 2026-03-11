from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
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

def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize self-collisions.

  Returns the number of self-collisions detected by the specified contact sensor.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)

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
