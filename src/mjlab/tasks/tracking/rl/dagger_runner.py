"""DAgger (distillation) runner for Tracking Policy.

Follows TWIST OnPolicyDaggerRunner: teacher on privileged (critic) obs,
student on policy obs; PPO + KL(teacher || student) with annealing.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, is_dataclass
from typing import Any, cast

import torch
import wandb
from rsl_rl.env.vec_env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.dagger_ppo import DaggerPPO
from mjlab.tasks.tracking.rl.exporter import (
  attach_onnx_metadata,
  export_motion_policy_as_onnx,
)


def _get_policy_path(teacher_checkpoint: str) -> str:
  """Resolve teacher checkpoint path (absolute or relative to cwd)."""
  if os.path.isabs(teacher_checkpoint):
    return teacher_checkpoint
  return os.path.abspath(teacher_checkpoint)


class MotionTrackingDaggerRunner(OnPolicyRunner):
  """On-policy runner with DAgger: student learns from teacher on privileged obs."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    self._full_cfg = copy.deepcopy(train_cfg)
    cfg_for_alg = copy.deepcopy(train_cfg)
    if "actor" not in cfg_for_alg and "policy" in cfg_for_alg:
      cfg_for_alg["actor"] = copy.deepcopy(cfg_for_alg["policy"])
      cfg_for_alg["critic"] = copy.deepcopy(cfg_for_alg["policy"])
    super().__init__(env, cfg_for_alg, log_dir, device)
    self.registry_name = registry_name

    teacher_forward = (train_cfg.get("teacher_forward_checkpoint") or "").strip()
    teacher_backward = (train_cfg.get("teacher_backward_checkpoint") or "").strip()
    eval_student = train_cfg.get("eval_student", False)
    dagger_coef = float(train_cfg.get("dagger_coef", 0.1))
    dagger_coef_anneal_steps = int(train_cfg.get("dagger_coef_anneal_steps", 30_000))
    dagger_coef_min = float(train_cfg.get("dagger_coef_min", 0.01))

    if not eval_student and (not teacher_forward or not teacher_backward):
      raise ValueError(
        "DAgger (双 Teacher) 需要 teacher_forward_checkpoint 与 teacher_backward_checkpoint。"
      )

    teacher_forward_actor = self._build_teacher_actor()
    teacher_backward_actor = self._build_teacher_actor()
    if teacher_forward_actor is None or teacher_backward_actor is None:
      raise RuntimeError(
        "MotionTrackingDaggerRunner: 无法构建双 Teacher（请检查 obs_groups / policy 配置）。"
      )
    if not eval_student:
      for path_key, actor in (
        (teacher_forward, teacher_forward_actor),
        (teacher_backward, teacher_backward_actor),
      ):
        path = _get_policy_path(path_key)
        if not os.path.isfile(path):
          raise FileNotFoundError(f"Teacher checkpoint 不存在: {path}")
        self._load_teacher(actor, path)

    old_alg = self.alg
    # rsl_rl PPO expects policy (ActorCritic); it does not take actor/critic separately
    self.alg = DaggerPPO(
      teacher_forward_actor=teacher_forward_actor,
      teacher_backward_actor=teacher_backward_actor,
      dagger_coef=dagger_coef,
      eval_student=eval_student,
      dagger_coef_anneal_steps=dagger_coef_anneal_steps,
      dagger_coef_min=dagger_coef_min,
      policy=old_alg.policy,
      storage=old_alg.storage,
      device=self.device,
      num_learning_epochs=old_alg.num_learning_epochs,
      num_mini_batches=old_alg.num_mini_batches,
      clip_param=old_alg.clip_param,
      gamma=old_alg.gamma,
      lam=old_alg.lam,
      value_loss_coef=old_alg.value_loss_coef,
      entropy_coef=old_alg.entropy_coef,
      learning_rate=old_alg.learning_rate,
      max_grad_norm=old_alg.max_grad_norm,
      optimizer=old_alg.optimizer,
      use_clipped_value_loss=old_alg.use_clipped_value_loss,
      schedule=old_alg.schedule,
      desired_kl=old_alg.desired_kl,
      normalize_advantage_per_mini_batch=old_alg.normalize_advantage_per_mini_batch,
      multi_gpu_cfg=getattr(old_alg, "multi_gpu_cfg", None),
    )
    self.alg.rnd = getattr(old_alg, "rnd", None)
    self.alg.rnd_optimizer = getattr(old_alg, "rnd_optimizer", None)
    self.alg.symmetry = getattr(old_alg, "symmetry", None)

  def _get_student_policy(self):
    """Return the student policy module (ActorCritic); rsl_rl PPO uses .policy, not .get_policy()."""
    return getattr(self.alg, "policy", None) or getattr(self.alg, "actor", None)

  def _build_teacher_actor(self):
    """Build teacher policy (same architecture as student, input = critic obs)."""
    try:
      obs = self.env.get_observations()
      policy = self._get_student_policy()
      if policy is None:
        return None
      actor_class = type(policy)
      # Support both dict (e.g. from wandb) and dataclass (RslRlDaggerRunnerCfg)
      if hasattr(self._full_cfg, "get"):
        policy_cfg = self._full_cfg.get("actor") or self._full_cfg.get("policy")
      else:
        policy_cfg = getattr(self._full_cfg, "actor", None) or getattr(self._full_cfg, "policy", None)
      if policy_cfg is None:
        return None
      if is_dataclass(policy_cfg):
        policy_kw = asdict(cast(Any, policy_cfg))
      else:
        policy_kw = dict(policy_cfg) if isinstance(policy_cfg, dict) else {}
      policy_kw.pop("class_name", None)
      # ActorCritic expects obs_groups with keys "policy" and "critic"; teacher uses critic obs for both
      teacher_obs_groups = {"policy": ("critic",), "critic": ("critic",)}
      teacher_actor = actor_class(
        obs,
        teacher_obs_groups,
        self.env.num_actions,
        **policy_kw,
      ).to(self.device)
      return teacher_actor
    except Exception as e:
      print(f"[WARN] MotionTrackingDaggerRunner: build teacher failed: {e}")
      return None

  def _load_teacher(self, teacher_actor: torch.nn.Module, path: str) -> None:
    print("*" * 80)
    print(f"Loading teacher policy from {path} ...")
    loaded = torch.load(path, map_location=self.device, weights_only=False)
    if "actor_state_dict" in loaded:
      teacher_actor.load_state_dict(loaded["actor_state_dict"], strict=False)
    elif "model_state_dict" in loaded:
      teacher_actor.load_state_dict(loaded["model_state_dict"], strict=False)
    else:
      teacher_actor.load_state_dict(loaded, strict=False)
    print("*" * 80)

  def save(self, path: str, infos=None):
    """Save student model and training state; export ONNX."""
    super().save(path, infos)
    policy_path = path.split("model")[0]
    policy = self._get_student_policy()
    if getattr(policy, "actor_obs_normalization", False):
      normalizer = getattr(policy, "actor_obs_normalizer", None)
    else:
      normalizer = None
    use_wandb = (
      hasattr(self, "logger")
      and getattr(self.logger, "logger_type", None) == "wandb"
    )
    if use_wandb and wandb.run is not None:
      run_name = wandb.run.name
      filename = policy_path.split("/")[-2] + ".onnx"
    else:
      run_name = "local"
      filename = "policy.onnx"
    onnx_path = os.path.join(policy_path, filename)
    export_motion_policy_as_onnx(
      self.env.unwrapped,
      policy,
      normalizer=normalizer,
      path=policy_path,
      filename=filename,
    )
    print(f"[INFO] ONNX exported locally: {onnx_path}")
    if use_wandb and wandb.run is not None:
      attach_onnx_metadata(
        self.env.unwrapped,
        run_name,
        path=policy_path,
        filename=filename,
      )
      wandb.save(onnx_path, base_path=os.path.dirname(policy_path))
      if self.registry_name is not None:
        wandb.run.use_artifact(self.registry_name)
        self.registry_name = None
