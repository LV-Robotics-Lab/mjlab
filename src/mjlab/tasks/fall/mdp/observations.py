from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


##
# Privileged (base/root and COM) for critic only.
##


def base_pos_rel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Base (root) position in world frame relative to env origin. Shape (num_envs, 3)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w - env.scene.env_origins


def com_pos_rel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Center-of-mass position in world frame relative to env origin. Shape (num_envs, 3)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_com_pos_w - env.scene.env_origins


def com_lin_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Center-of-mass linear velocity in world frame. Shape (num_envs, 3)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_com_lin_vel_w


##
# Joint state command (for joint_state command term).
##


def joint_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Target joint positions and velocities from the joint state command.

  Returns shape (num_envs, num_joints * 2): [joint_pos_cmd, joint_vel_cmd].
  """
  from mjlab.tasks.fall.mdp.commands import JointStateCommand

  cmd = cast(JointStateCommand, env.command_manager.get_term(command_name))
  return torch.cat([cmd.joint_pos, cmd.joint_vel], dim=1)


def joint_pos_error_from_command(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Joint position tracking error: current - target. Shape (num_envs, num_joints)."""
  from mjlab.tasks.fall.mdp.commands import JointStateCommand

  cmd = cast(JointStateCommand, env.command_manager.get_term(command_name))
  return cmd.robot_joint_pos - cmd.joint_pos


def joint_vel_error_from_command(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Joint velocity tracking error: current - target. Shape (num_envs, num_joints)."""
  from mjlab.tasks.fall.mdp.commands import JointStateCommand

  cmd = cast(JointStateCommand, env.command_manager.get_term(command_name))
  return cmd.robot_joint_vel - cmd.joint_vel
