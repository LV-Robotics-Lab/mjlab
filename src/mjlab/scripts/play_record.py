"""Record fall-recovery success rates under scripted push disturbances."""

from __future__ import annotations

import csv
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


@dataclass(frozen=True)
class PlayRecordConfig:
  wandb_run_path: str | None = None
  checkpoint_file: str | None = None
  motion_file: str | None = None
  device: str | None = None
  num_envs: int = 100
  push_time_s: float = 3.0
  eval_after_push_s: float = 8.0
  speed_min: float = 1.5
  speed_max: float = 4.0
  speed_step: float = 0.5
  output_csv: str = "logs/play_record/recovery_success_rates.csv"


def _build_policy(task: str, env, cfg: PlayRecordConfig, device: str):
  agent_cfg = load_rl_cfg(task)
  log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
  if cfg.checkpoint_file is not None:
    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
  else:
    if cfg.wandb_run_path is None:
      raise ValueError("Provide either `checkpoint_file` or `wandb_run_path`.")
    resume_path, was_cached = get_wandb_checkpoint_path(log_root_path, Path(cfg.wandb_run_path))
    print(
      f"[INFO] Loading checkpoint: {resume_path.name} "
      f"(run: {resume_path.parent.name}, {'cached' if was_cached else 'downloaded'})"
    )

  runner_cfg = asdict(agent_cfg)
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    runner_cls = OnPolicyRunner

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

  runner = runner_cls(cast(Any, env), runner_cfg, log_dir=None, device=device)
  runner.load(str(resume_path))
  return runner.get_inference_policy(device=device)


def _resolve_tracking_motion(task: str, env_cfg, cfg: PlayRecordConfig) -> None:
  is_tracking_task = (
    env_cfg.commands is not None
    and "motion" in env_cfg.commands
    and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
  )
  if not is_tracking_task:
    return
  assert env_cfg.commands is not None
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if cfg.motion_file is not None:
    motion_cmd.motion_file = cfg.motion_file
    return
  if cfg.wandb_run_path is None:
    raise ValueError("Tracking task requires `motion_file` or `wandb_run_path`.")
  import wandb

  wandb_run = wandb.Api().run(str(cfg.wandb_run_path))
  art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
  if art is None:
    raise RuntimeError("No motion artifact found in the run.")
  motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")


def _speed_values(cfg: PlayRecordConfig) -> list[float]:
  values: list[float] = []
  x = cfg.speed_min
  while x <= cfg.speed_max + 1e-6:
    values.append(round(x, 6))
    x += cfg.speed_step
  return values


def _evaluate_success(env, failed_mask: torch.Tensor) -> torch.Tensor:
  robot = env.unwrapped.scene["robot"]
  base_height = robot.data.root_link_pos_w[:, 2]
  projected_gravity_z = robot.data.projected_gravity_b[:, 2]
  joint_err_l2 = torch.linalg.norm(robot.data.joint_pos - robot.data.default_joint_pos, dim=1)

  # Tuned to match "stood up and returned close to default pose".
  stood_up = base_height > 0.72
  upright = projected_gravity_z < -0.9
  return stood_up & upright & (~failed_mask)


def run_record(task: str, cfg: PlayRecordConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task, play=True)
  _resolve_tracking_motion(task, env_cfg, cfg)
  if env_cfg.events is not None:
    env_cfg.events.pop("reset_base", None)
  env_cfg.scene.num_envs = int(cfg.num_envs)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  agent_cfg = load_rl_cfg(task)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  policy = _build_policy(task, env, cfg, device)

  step_dt = float(env.unwrapped.step_dt)
  push_step = max(1, int(round(cfg.push_time_s / step_dt)))
  eval_step = max(push_step + 1, int(round((cfg.push_time_s + cfg.eval_after_push_s) / step_dt)))
  push_env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.long)
  robot = env.unwrapped.scene["robot"]

  direction_map: list[tuple[str, tuple[float, float]]] = [
    ("front", (1.0, 0.0)),
    ("left", (0.0, 1.0)),
    ("back", (-1.0, 0.0)),
  ]
  speeds = _speed_values(cfg)
  results: list[dict[str, float | int | str]] = []

  with torch.inference_mode():
    for direction_name, (dx, dy) in direction_map:
      for speed in speeds:
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
          obs = reset_out[0]
        else:
          obs = reset_out
        failed_mask = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
        for step in range(1, eval_step + 1):
          if step == push_step:
            vel_w = robot.data.root_link_vel_w[push_env_ids].clone()
            vel_w[:, 0] += dx * speed
            vel_w[:, 1] += dy * speed
            robot.write_root_link_velocity_to_sim(vel_w, env_ids=push_env_ids)

          actions = policy(obs)
          obs, _, dones, _ = env.step(actions)
          if isinstance(dones, torch.Tensor) and dones.any():
            failed_mask |= dones.bool()

        success = _evaluate_success(env, failed_mask)
        success_count = int(success.sum().item())
        total = int(env.unwrapped.num_envs)
        success_rate = float(success_count / total)
        results.append(
          {
            "direction": direction_name,
            "speed_mps": speed,
            "num_envs": total,
            "num_success": success_count,
            "success_rate": success_rate,
          }
        )
        print(
          f"[RESULT] direction={direction_name:>5} speed={speed:.2f} "
          f"success={success_count}/{total} ({success_rate:.3f})"
        )

  output_path = Path(cfg.output_csv)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
      f, fieldnames=["direction", "speed_mps", "num_envs", "num_success", "success_rate"]
    )
    writer.writeheader()
    writer.writerows(results)
  print(f"[DONE] Wrote CSV: {output_path}")
  env.close()


def main() -> None:
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )
  args = tyro.cli(
    PlayRecordConfig,
    args=remaining_args,
    default=PlayRecordConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )
  run_record(chosen_task, args)


if __name__ == "__main__":
  main()
