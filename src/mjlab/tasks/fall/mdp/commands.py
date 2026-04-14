"""Joint state command for fall task.

Provides a command term that samples target joint positions and velocities
(e.g. from default + offset). The policy can track this command to learn
to hold or track joint state. Reference: tracking/mdp/commands.py (MotionCommand).
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class JointStateCommand(CommandTerm):
  """Command term that outputs target joint positions and velocities.

  On resample, samples joint_pos and joint_vel (default + random offset),
  optionally writes robot state to sim so the episode starts at the command.
  command property returns [joint_pos, joint_vel] concatenated.
  """

  cfg: JointStateCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: JointStateCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.asset_name]
    joint_ids, _ = self.robot.find_joints(cfg.joint_names)
    self._joint_ids = torch.tensor(
      joint_ids, device=env.device, dtype=torch.long
    )
    self._num_joints = len(joint_ids)

    default_pos = self.robot.data.default_joint_pos
    default_vel = self.robot.data.default_joint_vel
    assert default_pos is not None and default_vel is not None

    self.joint_pos_cmd = default_pos[:, self._joint_ids].clone()
    self.joint_vel_cmd = default_vel[:, self._joint_ids].clone()

    self.metrics["error_joint_pos"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_joint_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos_cmd, self.joint_vel_cmd], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.joint_pos_cmd

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.joint_vel_cmd

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos[:, self._joint_ids]

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel[:, self._joint_ids]

  def _update_metrics(self) -> None:
    self.metrics["error_joint_pos"] = torch.norm(
      self.robot_joint_pos - self.joint_pos_cmd, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.robot_joint_vel - self.joint_vel_cmd, dim=-1
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    default_pos = self.robot.data.default_joint_pos
    default_vel = self.robot.data.default_joint_vel
    soft_limits = self.robot.data.soft_joint_pos_limits
    assert default_pos is not None and default_vel is not None
    assert soft_limits is not None

    pos = default_pos[env_ids][:, self._joint_ids].clone()
    pos += sample_uniform(
      self.cfg.joint_position_range[0],
      self.cfg.joint_position_range[1],
      (len(env_ids), self._num_joints),
      self.device,
    )
    pos_lim = soft_limits[env_ids][:, self._joint_ids]
    pos = pos.clamp(pos_lim[..., 0], pos_lim[..., 1])
    self.joint_pos_cmd[env_ids] = pos

    vel = default_vel[env_ids][:, self._joint_ids].clone()
    vel += sample_uniform(
      self.cfg.joint_velocity_range[0],
      self.cfg.joint_velocity_range[1],
      (len(env_ids), self._num_joints),
      self.device,
    )
    self.joint_vel_cmd[env_ids] = vel

    if self.cfg.write_robot_state_on_resample:
      self.robot.write_joint_state_to_sim(
        pos, vel, env_ids=env_ids, joint_ids=self._joint_ids
      )

  def _update_command(self) -> None:
    # Static command: no change until next resample.
    pass


@dataclass(kw_only=True)
class JointStateCommandCfg(CommandTermCfg):
  """Configuration for joint state command."""

  asset_name: str = "robot"
  joint_names: tuple[str, ...] = (".*",)
  joint_position_range: tuple[float, float] = (-0.12, 0.12)
  joint_velocity_range: tuple[float, float] = (-0.05, 0.05)
  write_robot_state_on_resample: bool = True
  resampling_time_range: tuple[float, float] = (5.0, 10.0)
  class_type: type[CommandTerm] = JointStateCommand


class ResetUpwardForceCommand(CommandTerm):
  """Per-env upward force command used by reset assist."""

  cfg: "ResetUpwardForceCommandCfg"
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: "ResetUpwardForceCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._command = torch.zeros(self.num_envs, 1, device=self.device)
    self._command[:, 0] = float(cfg.force)
    self.metrics["command_mean"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _update_metrics(self) -> None:
    self.metrics["command_mean"][:] = self._command[:, 0]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Keep command persistent; curriculum updates it explicitly per-env.
    del env_ids

  def _update_command(self) -> None:
    pass


@dataclass(kw_only=True)
class ResetUpwardForceCommandCfg(CommandTermCfg):
  """Configuration for reset upward force command."""

  force: float
  resampling_time_range: tuple[float, float] = (100.0, 100.0)
  class_type: type[CommandTerm] = ResetUpwardForceCommand
