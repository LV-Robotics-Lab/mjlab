from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def tracking_recovery_curriculum(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  *,
  event_name: str,
  recovery_start_common_step: int,
  recovery_push_velocity_scale: float,
) -> dict[str, torch.Tensor]:
  """Enable recovery mode after a training step threshold.

  - Before threshold: keep original push strength and terminate as before.
  - After threshold: enable `env.recovery_enabled` and increase `push_robot`
    velocity_range by a scale factor.

  Note: CurriculumManager only recomputes on env reset, so this flag applies
  to episodes that start after the threshold.
  """
  del env_ids  # CurriculumManager passes env_ids, but this curriculum is global.

  # Ensure base velocity_range snapshot exists.
  event_term_cfg = env.event_manager.get_term_cfg(event_name)
  velocity_range = event_term_cfg.params["velocity_range"]

  if not hasattr(env, "_tracking_recovery_base_velocity_range"):
    # Store original min/max so we can scale up/down deterministically.
    env._tracking_recovery_base_velocity_range = dict(velocity_range)

  base_velocity_range = env._tracking_recovery_base_velocity_range

  recovery_enabled = env.common_step_counter >= recovery_start_common_step
  env.recovery_enabled = bool(recovery_enabled)

  for key in ("x", "y", "z", "roll", "pitch", "yaw"):
    base_min, base_max = base_velocity_range[key]
    if recovery_enabled:
      velocity_range[key] = (base_min * recovery_push_velocity_scale, base_max * recovery_push_velocity_scale)
    else:
      velocity_range[key] = (base_min, base_max)

  return {
    "recovery_enabled": torch.tensor(float(recovery_enabled), device=env.device),
    "push_scale": torch.tensor(float(recovery_push_velocity_scale), device=env.device),
    "common_step_counter": torch.tensor(float(env.common_step_counter), device=env.device),
  }

