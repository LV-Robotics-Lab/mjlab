from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.fall.mdp.events import reset_root_state_mixed
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class VisualizeDataResetConfig:
  task: str = "Mjlab-Falling-Flat-PM1-AMP"
  device: str | None = None
  seed: int = 0
  viewer: Literal["native", "viser"] = "native"
  keep_push: bool = False


def _validate_task(task: str) -> None:
  if task not in list_tasks():
    raise ValueError(f"Unknown task '{task}'.")


def _sample_one_data_reset_state(env: ManagerBasedRlEnv, keep_push: bool) -> None:
  reset_term = env.cfg.events.get("reset_base")
  if reset_term is None:
    raise RuntimeError("Environment has no reset_base event.")
  params = dict(reset_term.params)

  params["data_probability"] = 1.0
  params["tilt_pose_range"] = {}
  params["tilt_velocity_range"] = {}
  params["tilt_joint_position_range"] = (0.0, 0.0)
  params["tilt_joint_velocity_range"] = (0.0, 0.0)

  motion_files = params.get("motion_files", ())
  if not motion_files:
    raise RuntimeError(
      "reset_base.motion_files is empty. Please set fall data csv paths first."
    )

  env_ids = torch.tensor([0], dtype=torch.int64, device=env.device)
  reset_root_state_mixed(env=env, env_ids=env_ids, **params)

  if not keep_push and "push_at_reset" in env.cfg.events:
    push_term = env.cfg.events["push_at_reset"]
    velocity_range = push_term.params.get("velocity_range", {})
    zero_velocity_range = {k: (0.0, 0.0) for k in velocity_range.keys()}
    push_term.params["velocity_range"] = zero_velocity_range

  env.scene.write_data_to_sim()
  env.sim.forward()


def run_visualize_data_reset(cfg: VisualizeDataResetConfig) -> None:
  configure_torch_backends()
  _validate_task(cfg.task)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task, play=False)
  env_cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  _obs, _extras = env.reset(seed=cfg.seed)
  del _obs, _extras

  _sample_one_data_reset_state(env, keep_push=cfg.keep_push)

  agent_cfg = load_rl_cfg(cfg.task)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  action_shape = vec_env.unwrapped.action_space.shape  # type: ignore[attr-defined]

  class ZeroPolicy:
    def __call__(self, obs) -> torch.Tensor:
      del obs
      return torch.zeros(action_shape, device=vec_env.unwrapped.device)

  policy = ZeroPolicy()
  if cfg.viewer == "native":
    NativeMujocoViewer(vec_env, policy).run()
  else:
    ViserPlayViewer(vec_env, policy).run()
  env.close()


def main() -> None:
  import mjlab.tasks  # noqa: F401

  cfg = tyro.cli(
    VisualizeDataResetConfig,
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )
  run_visualize_data_reset(cfg)


if __name__ == "__main__":
  main()
