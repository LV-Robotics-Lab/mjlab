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

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    start_it = self.current_learning_iteration
    self._export_onnx_at_iter = start_it + num_learning_iterations - 1
    super().learn(num_learning_iterations, init_at_random_ep_len)

  def save(self, path: str, infos=None):
    """Save the model and training information."""
    super().save(path, infos)
    if getattr(self, "_export_onnx_at_iter", None) != self.current_learning_iteration:
      return
    logger = getattr(self, "logger", None)
    if getattr(logger, "logger_type", None) == "wandb":
      policy_path = path.split("model")[0]
      filename = policy_path.split("/")[-2] + ".onnx"
      if self.alg.policy.actor_obs_normalization:
        normalizer = self.alg.policy.actor_obs_normalizer
      else:
        normalizer = None
      export_motion_policy_as_onnx(
        self.env.unwrapped,
        self.alg.policy,
        normalizer=normalizer,
        path=policy_path,
        filename=filename,
      )
      attach_onnx_metadata(
        self.env.unwrapped,
        wandb.run.name,  # type: ignore
        path=policy_path,
        filename=filename,
      )
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

      # link the artifact registry to this run
      if self.registry_name is not None:
        wandb.run.use_artifact(self.registry_name)  # type: ignore
        self.registry_name = None
