from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def illegal_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return torch.any(sensor.data.found, dim=-1)


def bad_base_pos_z_only(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  threshold: float = 0.25,
) -> torch.Tensor:
  """Terminate when base (root) z is below env_origin.z - threshold (robot fell / base too low)."""
  asset: Entity = env.scene[asset_cfg.name]
  base_z = asset.data.root_link_pos_w[:, 2]
  origin_z = env.scene.env_origins[:, 2]
  return base_z < origin_z - threshold


def bad_anchor_ori(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  threshold: float = 0.8,
) -> torch.Tensor:
  """Terminate when projected gravity z in base frame deviates from upright.

  Upright: g_b[:, 2] = -1. Terminate when (1 + g_b[:, 2]) > threshold, i.e. tilted too much.
  """
  asset: Entity = env.scene[asset_cfg.name]
  g_b = quat_apply_inverse(
    asset.data.root_link_quat_w,
    asset.data.gravity_vec_w,
  )
  return (1.0 + g_b[:, 2]) > threshold


def bad_body_pos_z_only(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  body_names: tuple[str, ...] = (),
  threshold: float = 0.25,
) -> torch.Tensor:
  """Terminate when any of the given bodies has z below env_origin.z - threshold."""
  if not body_names:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  asset: Entity = env.scene[asset_cfg.name]
  body_ids, _ = asset.find_bodies(body_names, preserve_order=True)
  if not body_ids:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  body_pos_w = asset.data.body_link_pos_w  # (num_envs, num_bodies, 3)
  body_z = body_pos_w[:, body_ids, 2]  # (num_envs, len(body_ids))
  origin_z = env.scene.env_origins[:, 2:3]
  return torch.any(body_z < origin_z - threshold, dim=-1)

def bad_body_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  """Terminate when any named body is in contact according to a contact sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None

  if not body_names:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=sensor.data.found.device)

  slot_body_names = []
  seen = set()
  for slot in sensor._slots:
    if slot.primary_name not in seen:
      slot_body_names.append(slot.primary_name)
      seen.add(slot.primary_name)

  body_to_index = {name: i for i, name in enumerate(slot_body_names)}
  selected_indexes = [body_to_index[name] for name in body_names if name in body_to_index]
  if not selected_indexes:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=sensor.data.found.device)

  found = sensor.data.found[:, selected_indexes]
  return torch.any(found > 0, dim=-1)


def bad_body_contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  body_names: tuple[str, ...],
  force_threshold: float,
) -> torch.Tensor:
  """Terminate when any named body contact-force norm exceeds threshold."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None

  if not body_names:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=sensor.data.force.device)

  slot_body_names = []
  seen = set()
  for slot in sensor._slots:
    if slot.primary_name not in seen:
      slot_body_names.append(slot.primary_name)
      seen.add(slot.primary_name)

  body_to_index = {name: i for i, name in enumerate(slot_body_names)}
  selected_indexes = [body_to_index[name] for name in body_names if name in body_to_index]
  if not selected_indexes:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=sensor.data.force.device)

  selected_force = sensor.data.force[:, selected_indexes]  # [B, K, 3]
  selected_force_norm = torch.norm(selected_force, dim=-1)  # [B, K]
  max_selected_force = torch.max(selected_force_norm, dim=-1)[0]  # [B]
  return max_selected_force > force_threshold