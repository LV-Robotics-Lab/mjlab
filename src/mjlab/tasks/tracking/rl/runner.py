import os

import wandb
from rsl_rl.env.vec_env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.rl.exporter import (
  attach_onnx_metadata,
  export_motion_policy_as_onnx,
)


class MotionTrackingOnPolicyRunner(OnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name

  def save(self, path: str, infos=None):
    """Save the model and training information."""
    super().save(path, infos)
    
    # Always export ONNX locally
    policy_path = path.split("model")[0]
    if self.alg.policy.actor_obs_normalization:
      normalizer = self.alg.policy.actor_obs_normalizer
    else:
      normalizer = None
    
    # Determine filename - use run name if available, otherwise use directory name
    use_wandb = hasattr(self, 'logger') and hasattr(self.logger, 'logger_type') and self.logger.logger_type in ["wandb"]
    if use_wandb and wandb.run is not None:
      run_name = wandb.run.name
      filename = policy_path.split("/")[-2] + ".onnx"
    else:
      run_name = "local"
      filename = "policy.onnx"
    
    # Export ONNX to local path
    onnx_path = os.path.join(policy_path, filename)
    export_motion_policy_as_onnx(
      self.env.unwrapped,
      self.alg.policy,
      normalizer=normalizer,
      path=policy_path,
      filename=filename,
    )
    print(f"[INFO] ONNX exported locally: {onnx_path}")
    
    # Wandb-specific: attach metadata and upload
    if use_wandb and wandb.run is not None:
      attach_onnx_metadata(
        self.env.unwrapped,
        run_name,
        path=policy_path,
        filename=filename,
      )
      wandb.save(onnx_path, base_path=os.path.dirname(policy_path))

      # link the artifact registry to this run
      if self.registry_name is not None:
        wandb.run.use_artifact(self.registry_name)
        self.registry_name = None
