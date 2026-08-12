"""Stress-test SceneEntityCfg ID indexing with a synthetic many-term env.

Builds a cartpole env inflated with N extra observation and reward terms, each
using a single-joint ``SceneEntityCfg`` so ``joint_ids`` resolves to a real
``list[int]`` (not the optimized ``slice(None)``). This is a focused microbench
for the ``tensor[:, cfg.joint_ids]`` implicit-CUDA-sync question raised in
issue #1019: light physics keeps manager overhead as the dominant signal, and
every extra term hits the suspected hot path.

Reuses the timing helpers from ``measure_throughput`` for apples-to-apples
comparison with the regression benchmark.
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.cartpole.cartpole_env_cfg import (
  cartpole_balance_env_cfg,
  pole_angle_cos_sin,
)

sys.path.insert(0, str(Path(__file__).parent))
from measure_throughput import measure_env_sps, measure_physics_sps  # noqa: E402


def _synthetic_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Indexing-heavy reward that mirrors the joint_pos_rel access pattern."""
  asset = env.scene[asset_cfg.name]
  jp = asset.data.joint_pos[:, asset_cfg.joint_ids]
  jv = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return -(jp.pow(2).sum(-1) + 0.1 * jv.pow(2).sum(-1))


def _inflate(cfg, extra_terms: int) -> None:
  """Add ``extra_terms`` obs + reward terms that all do list[int] indexing.

  Alternates between the slider and hinge joints; each picks exactly one joint
  so ``joint_ids`` cannot collapse to ``slice(None)``.
  """
  if extra_terms <= 0:
    return
  joint_names = ("slider", "hinge_1")
  obs_funcs = (joint_pos_rel, joint_vel_rel, pole_angle_cos_sin)
  for i in range(extra_terms):
    joint = joint_names[i % 2]
    entity_cfg = SceneEntityCfg("cartpole", joint_names=(joint,))
    obs_func = obs_funcs[i % len(obs_funcs)]
    cfg.observations["actor"].terms[f"synth_obs_{i}"] = ObservationTermCfg(
      func=obs_func,
      params={"asset_cfg": entity_cfg},
    )
    cfg.observations["critic"].terms[f"synth_obs_{i}"] = ObservationTermCfg(
      func=obs_func,
      params={"asset_cfg": entity_cfg},
    )
    cfg.rewards[f"synth_reward_{i}"] = RewardTermCfg(
      func=_synthetic_reward,
      weight=0.0,
      params={"asset_cfg": entity_cfg},
    )


@dataclass
class SyntheticConfig:
  num_envs: int = 4096
  num_steps: int = 200
  warmup_steps: int = 50
  reps: int = 5
  device: str = "cuda:0"
  extra_terms: list[int] = field(default_factory=lambda: [0, 10, 50, 100])
  """Sweep of extra-term counts to benchmark."""


def benchmark_one(cfg: SyntheticConfig, extra_terms: int) -> dict:
  print(f"\nSynthetic cartpole with {extra_terms} extra terms...")
  env_cfg = cartpole_balance_env_cfg()
  env_cfg.scene.num_envs = cfg.num_envs
  _inflate(env_cfg, extra_terms)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  env.reset()

  action_dim = sum(env.action_manager.action_term_dim)
  action = torch.zeros(env.num_envs, action_dim, device=env.device)
  for _ in range(cfg.warmup_steps):
    env.step(action)
  torch.cuda.synchronize()

  physics_samples: list[float] = []
  env_samples: list[float] = []
  for _ in range(cfg.reps):
    physics_samples.append(measure_physics_sps(env, cfg.num_steps))
    env.reset()
    torch.cuda.synchronize()
    env_samples.append(measure_env_sps(env, cfg.num_steps))
    env.reset()
    torch.cuda.synchronize()

  env.close()

  physics_med = statistics.median(physics_samples)
  env_med = statistics.median(env_samples)
  return {
    "extra_terms": extra_terms,
    "physics_sps": physics_med,
    "env_sps": env_med,
    "overhead_pct": 100 * (1 - env_med / physics_med),
    "env_sps_min": min(env_samples),
    "env_sps_max": max(env_samples),
  }


def main(cfg: SyntheticConfig) -> None:
  print("Synthetic Many-Term Throughput Benchmark (cartpole base)")
  print(f"  Envs: {cfg.num_envs}  Steps: {cfg.num_steps}  Reps: {cfg.reps}")

  results = [benchmark_one(cfg, n) for n in cfg.extra_terms]

  print("\n" + "=" * 72)
  print(
    f"{'Extra terms':>12} {'Physics SPS':>14} {'Env SPS':>14} "
    f"{'Env min/max':>20} {'Overhead':>10}"
  )
  print("-" * 72)
  for r in results:
    minmax = f"{r['env_sps_min']:,.0f}/{r['env_sps_max']:,.0f}"
    print(
      f"{r['extra_terms']:>12} {r['physics_sps']:>14,.0f} "
      f"{r['env_sps']:>14,.0f} {minmax:>20} {r['overhead_pct']:>9.1f}%"
    )


if __name__ == "__main__":
  cfg = tyro.cli(SyntheticConfig, config=mjlab.TYRO_FLAGS)
  main(cfg)
