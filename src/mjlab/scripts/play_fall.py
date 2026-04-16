"""Play script with post-reset push disturbance for fall testing."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayFallConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"

  # Root velocity kick (world-frame): random direction in xy, added to floating base.
  # PM1 root is LINK_BASE; whole upper body (incl. torso) moves with this kick.
  # First kick happens after this interval, then repeats every interval.
  push_interval_s: float = 8.0
  push_root_lin_vel_xy_min: float = 2.5
  push_root_lin_vel_xy_max: float = 4.5

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


class TimedRootLinVelKickWrapper:
  """Add random world-frame xy root velocity every N steps since reset."""

  def __init__(self, env, cfg: PlayFallConfig):
    self.env = env
    self._unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    self._device = self._unwrapped.device
    self._num_envs = self._unwrapped.num_envs
    self._robot = self._unwrapped.scene["robot"]

    step_dt = float(getattr(self._unwrapped, "step_dt", 0.02))
    self._interval_steps = max(1, int(round(cfg.push_interval_s / step_dt)))
    self._lin_vel_min = float(cfg.push_root_lin_vel_xy_min)
    self._lin_vel_max = float(cfg.push_root_lin_vel_xy_max)

    self._steps_since_reset = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)

    print(
      "[INFO] Timed root xy velocity kick: "
      f"interval={cfg.push_interval_s:.2f}s (~every {self._interval_steps} steps), "
      f"|v_xy| in [{self._lin_vel_min:.2f}, {self._lin_vel_max:.2f}] m/s (world frame)."
    )

  def __getattr__(self, name):
    return getattr(self.env, name)

  def _apply_velocity_kick_if_needed(self) -> None:
    trigger_env_ids = torch.nonzero(
      (self._steps_since_reset > 0)
      & (self._steps_since_reset % self._interval_steps == 0),
      as_tuple=False,
    ).squeeze(-1)
    if trigger_env_ids.numel() == 0:
      return
    vel_w = self._robot.data.root_link_vel_w[trigger_env_ids].clone()
    angles = 2.0 * torch.pi * torch.rand(trigger_env_ids.numel(), device=self._device)
    magnitudes = torch.empty(trigger_env_ids.numel(), device=self._device).uniform_(
      self._lin_vel_min, self._lin_vel_max
    )
    dvx = torch.cos(angles) * magnitudes
    dvy = torch.sin(angles) * magnitudes
    vel_w[:, 0] += dvx
    vel_w[:, 1] += dvy
    self._robot.write_root_link_velocity_to_sim(vel_w, env_ids=trigger_env_ids)
    for env_id, dx, dy in zip(trigger_env_ids.tolist(), dvx.tolist(), dvy.tolist()):
      print(
        f"[KICK] env={env_id} root d(vx,vy)=({dx:+.3f}, {dy:+.3f}) m/s "
        f"(step_since_reset={int(self._steps_since_reset[env_id].item())})"
      )

  def step(self, actions):
    self._apply_velocity_kick_if_needed()
    obs, rew, dones, extras = self.env.step(actions)
    self._steps_since_reset += 1
    if isinstance(dones, torch.Tensor) and dones.any():
      done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
      self._steps_since_reset[done_ids] = 0
    return obs, rew, dones, extras

  def reset(self, *args, **kwargs):
    obs = self.env.reset(*args, **kwargs)
    self._steps_since_reset.zero_()
    return obs


def run_play(task: str, cfg: PlayFallConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task, play=True)
  agent_cfg = load_rl_cfg(task)
  # No random mixed reset: remove reset_base so only scene/entity defaults apply.
  if env_cfg.events is not None:
    env_cfg.events.pop("reset_base", None)

  dummy_mode = cfg.agent in {"zero", "random"}
  trained_mode = not dummy_mode

  is_tracking_task = (
    env_cfg.commands is not None
    and "motion" in env_cfg.commands
    and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
  )

  if is_tracking_task and cfg._demo_mode:
    assert env_cfg.commands is not None
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    assert env_cfg.commands is not None
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    if dummy_mode:
      if not cfg.registry_name:
        raise ValueError("Tracking tasks require `registry_name` when using dummy agents.")
      registry_name = cfg.registry_name if ":" in cfg.registry_name else f"{cfg.registry_name}:latest"
      import wandb

      artifact = wandb.Api().artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        motion_cmd.motion_file = cfg.motion_file
      elif cfg.wandb_run_path is not None:
        import wandb

        wandb_run = wandb.Api().run(str(cfg.wandb_run_path))
        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
          raise RuntimeError("No motion artifact found in the run.")
        motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if trained_mode:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError("`wandb_run_path` is required when `checkpoint_file` is not provided.")
      resume_path, was_cached = get_wandb_checkpoint_path(log_root_path, Path(cfg.wandb_run_path))
      run_id = resume_path.parent.name
      print(f"[INFO]: Loading checkpoint: {resume_path.name} (run: {run_id}, {'cached' if was_cached else 'downloaded'})")
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (trained_mode and cfg.video) else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if trained_mode and cfg.video:
    assert log_dir is not None
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play_fall",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  env = TimedRootLinVelKickWrapper(env, cfg)

  if dummy_mode:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore

    class Policy:
      def __call__(self, obs) -> torch.Tensor:
        del obs
        if cfg.agent == "zero":
          return torch.zeros(action_shape, device=env.unwrapped.device)
        return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

    policy = Policy()
  else:
    runner_cfg = asdict(agent_cfg)
    runner_cls = load_runner_cls(task)
    if runner_cls is None:
      runner_cls = MotionTrackingOnPolicyRunner if is_tracking_task else OnPolicyRunner
    alg_cfg = runner_cfg.get("algorithm") or {}
    alg_class_name = alg_cfg.get("class_name", "")
    if isinstance(alg_class_name, str) and "amp_rsl_rl" in alg_class_name:
      use_mjlab_amp_runner = getattr(runner_cls, "__name__", "") == "MjlabAmpOnPolicyRunner"
      runner_cfg["algorithm"] = {
        **alg_cfg,
        "class_name": (
          "mjlab.rl.mj_amp_ppo.MjlabAmpPPO" if use_mjlab_amp_runner else alg_cfg.get("class_name", "")
        ),
      }
      if "discriminator" not in runner_cfg:
        disc_hidden = alg_cfg.get("disc_hidden_dims", (1024, 512))
        runner_cfg["discriminator"] = {
          "hidden_dims": list(disc_hidden),
          "reward_scale": alg_cfg.get("disc_reward_scale", 2.0),
          "loss_type": "BCEWithLogits",
          "empirical_normalization": True,
        }
      if "dataset" not in runner_cfg:
        runner_cfg["dataset"] = {}

    runner = runner_cls(cast(Any, env), runner_cfg, log_dir=str(log_dir), device=device)
    runner.load(str(resume_path))
    policy = runner.get_inference_policy(device=device)

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(cast(Any, env), policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(cast(Any, env), policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )
  args = tyro.cli(
    PlayFallConfig,
    args=remaining_args,
    default=PlayFallConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )
  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
