"""Tracking-task MDP events (wrappers around generic env events)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.envs.mdp.events import push_by_setting_velocity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def push_by_setting_velocity_skip_recovery(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Like :func:`push_by_setting_velocity` but skips environments in recovery mode."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
  elif env_ids.numel() == 0:
    return

  recovery_buf = getattr(env, "recovery_mode_buf", None)
  if recovery_buf is not None:
    in_recovery = recovery_buf[env_ids]
    env_ids = env_ids[~in_recovery]
    if env_ids.numel() == 0:
      return

  push_by_setting_velocity(
    env, cast(torch.Tensor, env_ids), velocity_range, asset_cfg=asset_cfg
  )
