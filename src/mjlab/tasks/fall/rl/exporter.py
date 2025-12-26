import os
from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.tasks.fall.mdp import FallCommand
from mjlab.utils.lab_api.rl.exporter import _OnnxPolicyExporter


def export_fall_policy_as_onnx(
  env: ManagerBasedRlEnv,
  actor_critic: object,
  path: str,
  normalizer: object | None = None,
  filename="policy.onnx",
  verbose=False,
):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
  policy_exporter = _OnnxFallPolicyExporter(env, actor_critic, normalizer, verbose)
  policy_exporter.export(path, filename)


class _OnnxFallPolicyExporter(_OnnxPolicyExporter):
  def __init__(
    self, env: ManagerBasedRlEnv, actor_critic, normalizer=None, verbose=False
  ):
    super().__init__(actor_critic, normalizer, verbose)
    cmd = cast(FallCommand, env.command_manager.get_term("fall"))

    self.joint_pos = cmd.fall.joint_pos.to("cpu")
    self.joint_vel = cmd.fall.joint_vel.to("cpu")
    self.body_pos_w = cmd.fall.body_pos_w.to("cpu")
    self.body_quat_w = cmd.fall.body_quat_w.to("cpu")
    self.body_lin_vel_w = cmd.fall.body_lin_vel_w.to("cpu")
    self.body_ang_vel_w = cmd.fall.body_ang_vel_w.to("cpu")
    self.time_step_total: int = self.joint_pos.shape[0]

  def forward(self, x, time_step):  # pyright: ignore [reportIncompatibleMethodOverride]
    time_step_clamped = torch.clamp(
      time_step.long().squeeze(-1), max=self.time_step_total - 1
    )
    return (
      self.actor(self.normalizer(x)),
      self.joint_pos[time_step_clamped],
      self.joint_vel[time_step_clamped],
      self.body_pos_w[time_step_clamped],
      self.body_quat_w[time_step_clamped],
      self.body_lin_vel_w[time_step_clamped],
      self.body_ang_vel_w[time_step_clamped],
    )

  def export(self, path, filename):
    self.to("cpu")
    obs = torch.zeros(1, self.actor[0].in_features)
    time_step = torch.zeros(1, 1)
    torch.onnx.export(
      self,
      (obs, time_step),
      os.path.join(path, filename),
      export_params=True,
      opset_version=11,
      verbose=self.verbose,
      input_names=["obs", "time_step"],
      output_names=[
        "actions",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      ],
      dynamic_axes={},
      dynamo=False,
    )


def attach_onnx_metadata(
  env: ManagerBasedRlEnv, run_path: str, path: str, filename="policy.onnx"
) -> None:
  """Attach fall-specific metadata to ONNX model.

  Args:
    env: The RL environment.
    run_path: W&B run path or other identifier.
    path: Directory containing the ONNX file.
    filename: Name of the ONNX file.
  """
  onnx_path = os.path.join(path, filename)

  # Get base metadata common to all tasks.
  metadata = get_base_metadata(env, run_path)

  # Add fall-specific metadata.
  fall_term = env.command_manager.get_term("fall")
  assert isinstance(fall_term, FallCommand)
  fall_term_cfg = fall_term.cfg
  metadata.update(
    {
      "anchor_body_name": fall_term_cfg.anchor_body_name,
      "body_names": list(fall_term_cfg.body_names),
    }
  )

  attach_metadata_to_onnx(onnx_path, metadata)
