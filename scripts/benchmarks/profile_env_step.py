"""Profile env.step() to localize the cost of SceneEntityCfg indexing.

Wraps a small window of ``env.step()`` calls in ``torch.profiler`` and
surfaces:

- Top operators by CUDA self time.
- Per-step counts of ``aten::index`` / ``aten::index_put_`` — the operators
  that fire on every ``tensor[:, cfg.joint_ids]`` with a Python list index.
- Per-step count of host-to-device ``Memcpy`` events, a proxy for the
  list-of-int → CUDA-tensor copies issue #1019 is about.

Supports two modes:

- ``--task <registered task>`` (e.g. ``Mjlab-Tracking-Flat-Unitree-G1``).
  Pass ``--motion-file`` to skip wandb when the task needs one.
- ``--synthetic-extra-terms N``: cartpole inflated with N single-joint
  obs+reward terms (each with an explicit ``joint_ids`` list, defeating the
  ``slice(None)`` optimization). Use 0 as the control.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from torch.profiler import ProfilerActivity, profile

import mjlab
import mjlab.tasks  # noqa: F401 - registers tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg

sys.path.insert(0, str(Path(__file__).parent))
from measure_synthetic_throughput import _inflate  # noqa: E402


@dataclass
class ProfileConfig:
  task: str | None = None
  """Registered task name. Mutually exclusive with synthetic mode."""

  synthetic_extra_terms: int | None = None
  """If set, profile cartpole inflated with this many extra terms."""

  motion_file: str | None = None
  """Local path to motion.npz, used when task needs one (skips wandb)."""

  num_envs: int = 4096
  num_steps: int = 20
  """Number of env steps to profile. Keep small — profiler overhead is real."""

  warmup_steps: int = 50
  device: str = "cuda:0"
  trace_dir: Path = Path("profile_traces")
  top_n: int = 25


def _build_env(cfg: ProfileConfig) -> tuple[ManagerBasedRlEnv, str]:
  if (cfg.task is None) == (cfg.synthetic_extra_terms is None):
    raise SystemExit("Set exactly one of --task / --synthetic-extra-terms.")

  if cfg.synthetic_extra_terms is not None:
    label = f"synthetic-cartpole-{cfg.synthetic_extra_terms}"
    env_cfg = cartpole_balance_env_cfg()
    env_cfg.scene.num_envs = cfg.num_envs
    _inflate(env_cfg, cfg.synthetic_extra_terms)
  else:
    assert cfg.task is not None
    label = cfg.task
    env_cfg = load_env_cfg(cfg.task)
    env_cfg.scene.num_envs = cfg.num_envs
    motion_cmd = env_cfg.commands.get("motion") if len(env_cfg.commands) > 0 else None
    if isinstance(motion_cmd, MotionCommandCfg):
      if cfg.motion_file is None:
        raise SystemExit(
          f"Task {cfg.task} needs --motion-file (local motion.npz path)."
        )
      motion_cmd.motion_file = cfg.motion_file

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  return env, label


def _summarize(prof: profile, num_steps: int, top_n: int) -> None:
  events = prof.key_averages()

  print(f"\nTop {top_n} ops by CUDA self time:")
  print(
    events.table(
      sort_by="self_cuda_time_total",
      row_limit=top_n,
      max_name_column_width=55,
    )
  )

  def _count(name: str) -> int:
    return sum(int(e.count) for e in events if e.key == name)

  per_step_index = _count("aten::index") / num_steps
  per_step_index_put = _count("aten::index_put_") / num_steps
  per_step_to_copy = _count("aten::_to_copy") / num_steps
  per_step_index_select = _count("aten::index_select") / num_steps

  print("\nPer-step op counts (the implicit-sync suspects):")
  print(f"  aten::index         : {per_step_index:.1f}")
  print(f"  aten::index_put_    : {per_step_index_put:.1f}")
  print(f"  aten::index_select  : {per_step_index_select:.1f}")
  print(f"  aten::_to_copy      : {per_step_to_copy:.1f}")


def main(cfg: ProfileConfig) -> None:
  env, label = _build_env(cfg)
  env.reset()

  action_dim = sum(env.action_manager.action_term_dim)
  action = torch.zeros(env.num_envs, action_dim, device=env.device)
  for _ in range(cfg.warmup_steps):
    env.step(action)
  torch.cuda.synchronize()

  print(f"\nProfiling {label} for {cfg.num_steps} steps (num_envs={cfg.num_envs})...")

  with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
    with_stack=False,
  ) as prof:
    for _ in range(cfg.num_steps):
      env.step(action)
    torch.cuda.synchronize()

  cfg.trace_dir.mkdir(parents=True, exist_ok=True)
  trace_path = cfg.trace_dir / f"{label}.json"
  prof.export_chrome_trace(str(trace_path))
  print(f"Chrome trace: {trace_path}")

  _summarize(prof, cfg.num_steps, cfg.top_n)

  env.close()


if __name__ == "__main__":
  cfg = tyro.cli(ProfileConfig, config=mjlab.TYRO_FLAGS)
  main(cfg)
