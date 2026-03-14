"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlOnPolicyRunnerCfg
  registry_name: str | None = None
  motion_file: str | None = None
  teacher_forward_checkpoint: str | None = None
  """Path to 前摔 teacher checkpoint."""
  teacher_backward_checkpoint: str | None = None
  """Path to 后摔 teacher checkpoint."""
  motion_forward_file: str | None = None
  """前摔 motion .npz（双 Teacher distill 用）；也可在 pm1_distill_env_cfg(motion_forward_file=...) 提前指定。"""
  motion_backward_file: str | None = None
  """后摔 motion .npz（双 Teacher distill 用）；也可在 pm1_distill_env_cfg(motion_backward_file=...) 提前指定。"""
  map: bool = True
  """是否使用护具 map。传 --map false 则关闭。"""
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  # Set wandb entity and project early, before runner initialization
  if "WANDB_ENTITY" not in os.environ:
    os.environ["WANDB_ENTITY"] = "e1519767-national-university-of-singapore"
  if "WANDB_PROJECT" not in os.environ:
    os.environ["WANDB_PROJECT"] = cfg.agent.wandb_project
  
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  if not cfg.map and cfg.env.rewards and "reduce_contact_force" in cfg.env.rewards:
    cfg.env.rewards["reduce_contact_force"].params["protector_map_dir"] = None
    cfg.env.rewards["reduce_contact_force"].params["force_params_path"] = None
    if rank == 0:
      print("[INFO] 护具 map 已关闭（reduce_contact_force 不使用护具查表）")

  registry_name: str | None = None

  is_tracking_task = False  # 单 motion 的 tracking 任务（非 distill 双 motion）
  # 双 Teacher distill：motion_forward + motion_backward
  is_distill_fall_task = (
    cfg.env.commands is not None
    and "motion_forward" in cfg.env.commands
    and "motion_backward" in cfg.env.commands
    and isinstance(cfg.env.commands["motion_forward"], MotionCommandCfg)
    and isinstance(cfg.env.commands["motion_backward"], MotionCommandCfg)
  )

  if is_distill_fall_task:
    assert cfg.env.commands is not None
    motion_fwd = cfg.env.commands["motion_forward"]
    motion_bwd = cfg.env.commands["motion_backward"]
    assert isinstance(motion_fwd, MotionCommandCfg) and isinstance(motion_bwd, MotionCommandCfg)
    if cfg.motion_forward_file:
      p = Path(cfg.motion_forward_file)
      if not p.exists():
        raise FileNotFoundError(f"Motion forward file not found: {p}")
      motion_fwd.motion_file = str(p.resolve())
    if cfg.motion_backward_file:
      p = Path(cfg.motion_backward_file)
      if not p.exists():
        raise FileNotFoundError(f"Motion backward file not found: {p}")
      motion_bwd.motion_file = str(p.resolve())
    if not motion_fwd.motion_file or not motion_bwd.motion_file:
      raise ValueError(
        "双 Teacher distill 需要两个 motion 文件，请提供：\n"
        "  --motion-forward-file <前摔.npz> --motion-backward-file <后摔.npz>\n"
        "或在注册任务时用 pm1_distill_env_cfg(motion_forward_file=..., motion_backward_file=...) 指定。"
      )
    if rank == 0:
      print(f"[INFO] Motion forward: {motion_fwd.motion_file}")
      print(f"[INFO] Motion backward: {motion_bwd.motion_file}")
  else:
    is_tracking_task = (
      cfg.env.commands is not None
      and "motion" in cfg.env.commands
      and isinstance(cfg.env.commands["motion"], MotionCommandCfg)
    )
    if is_tracking_task:
      assert cfg.env.commands is not None
      motion_cmd = cfg.env.commands["motion"]
      assert isinstance(motion_cmd, MotionCommandCfg)

      if cfg.motion_file is not None:
        motion_file_path = Path(cfg.motion_file)
        if not motion_file_path.exists():
          raise FileNotFoundError(
            f"Motion file not found: {motion_file_path}\n"
            f"Please provide a valid path to the motion.npz file."
          )
        motion_cmd.motion_file = str(motion_file_path.resolve())
        if rank == 0:
          print(f"[INFO] Using motion file from CLI: {motion_cmd.motion_file}")
      elif cfg.registry_name is not None:
        # Download from wandb registry.
        registry_name = cast(str, cfg.registry_name)
        if ":" not in registry_name:
          registry_name = registry_name + ":latest"
        import wandb

        try:
          api = wandb.Api()
          if rank == 0:
            print(f"[INFO] Downloading motion artifact from wandb registry: {registry_name}")
          artifact = api.artifact(registry_name)
          motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
          if rank == 0:
            print(f"[INFO] Successfully downloaded motion file: {motion_cmd.motion_file}")
        except wandb.errors.CommError as e:
          error_msg = (
          f"Failed to download motion artifact from wandb registry: {registry_name}\n"
          f"Error: {e}\n\n"
          f"Possible solutions:\n"
          f"  1. Request access to the wandb registry from the project owner\n"
          f"  2. Download the motion file manually and use --motion-file <path>\n"
          f"  3. Verify your wandb authentication: wandb login"
          )
          raise RuntimeError(error_msg) from e
        except Exception as e:
          error_msg = (
          f"Unexpected error while downloading motion artifact: {registry_name}\n"
          f"Error: {e}\n\n"
          f"Consider using --motion-file <path> to specify a local motion file instead."
          )
          raise RuntimeError(error_msg) from e
      else:
        raise ValueError(
          "For tracking tasks, you must provide either:\n"
          "  --registry-name <wandb-registry-path>  (to download from wandb)\n"
          "  --motion-file <path-to-motion.npz>     (to use a local file)"
        )

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      # Load checkpoint from W&B.
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  if cfg.teacher_forward_checkpoint is not None and "teacher_forward_checkpoint" in agent_cfg:
    agent_cfg["teacher_forward_checkpoint"] = cfg.teacher_forward_checkpoint
  if cfg.teacher_backward_checkpoint is not None and "teacher_backward_checkpoint" in agent_cfg:
    agent_cfg["teacher_backward_checkpoint"] = cfg.teacher_backward_checkpoint
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = OnPolicyRunner

  runner_kwargs = {}
  if is_tracking_task:
    runner_kwargs["registry_name"] = registry_name

  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"
  
  # Set wandb entity and project for logging to e1519767-national-university-of-singapore/mjlab
  if "WANDB_ENTITY" not in os.environ:
    os.environ["WANDB_ENTITY"] = "e1519767-national-university-of-singapore"
  if "WANDB_PROJECT" not in os.environ:
    os.environ["WANDB_PROJECT"] = args.agent.wandb_project

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*", "WANDB*"),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
