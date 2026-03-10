from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import MotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )


def bad_anchor_ori(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )

  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )

  return (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  return torch.any(error > threshold, dim=-1)


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
