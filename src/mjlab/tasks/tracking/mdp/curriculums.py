from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class RewardWeightStage(TypedDict):
  """Stage configuration for reward weight curriculum.
  
  Attributes:
    step: Environment step at which to apply this weight.
    weight: Target weight value for the reward term.
  """
  step: int
  weight: float


class ThresholdStage(TypedDict):
  """Stage configuration for threshold curriculum.
  
  Attributes:
    step: Environment step at which to apply this threshold.
    threshold: Target threshold value.
  """
  step: int
  threshold: float


class VelocityRangeStage(TypedDict):
  """Stage configuration for velocity range curriculum.
  
  Attributes:
    step: Environment step at which to apply this velocity range.
    velocity_range: Dictionary with keys {"x", "y", "z", "roll", "pitch", "yaw"}
                   and values as (min, max) tuples.
  """
  step: int
  velocity_range: dict[str, tuple[float, float]]


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages.
  
  This function allows you to adjust reward weights dynamically during training.
  Stages are applied in order, so later stages override earlier ones when
  the training step exceeds the stage's step threshold.
  
  Args:
    env: The RL environment instance.
    env_ids: Environment indices (unused, kept for API consistency).
    reward_name: Name of the reward term to adjust (must exist in rewards dict).
    weight_stages: List of stages, each with 'step' (environment step) and 
                  'weight' (target weight). Stages should be ordered by step.
  
  Returns:
    Tensor containing the current weight value for logging.
  
  Example:
    weight_stages = [
      {"step": 0, "weight": 2.0},           # Start with weight 2.0
      {"step": 10000 * 24, "weight": 1.0},  # After 240k steps, reduce to 1.0
      {"step": 20000 * 24, "weight": 0.5},  # After 480k steps, reduce to 0.5
    ]
  """
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])


def event_velocity_range(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  velocity_stages: list[VelocityRangeStage],
) -> torch.Tensor:
  """Update an event's velocity range based on training step stages.
  
  This function allows you to adjust initial velocity ranges dynamically during training.
  Stages are applied in order, so later stages override earlier ones when
  the training step exceeds the stage's step threshold.
  
  Args:
    env: The RL environment instance.
    env_ids: Environment indices (unused, kept for API consistency).
    event_name: Name of the event term to adjust (must exist in events dict).
    velocity_stages: List of stages, each with 'step' (environment step) and
                    'velocity_range' (dict with velocity ranges). Stages should be ordered by step.
  
  Returns:
    Tensor containing the maximum velocity range magnitude for logging.
  
  Example:
    velocity_stages = [
      {"step": 0, "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}},
      {"step": 10000 * 24, "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
      {"step": 20000 * 24, "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    ]
  """
  del env_ids  # Unused.
  event_term_cfg = env.event_manager.get_term_cfg(event_name)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      # Update the velocity_range parameter in the event configuration
      event_term_cfg.params["velocity_range"] = stage["velocity_range"]
  
  # Return current max velocity range for logging
  current_range = event_term_cfg.params.get("velocity_range", {})
  max_range = 0.0
  if current_range:
    for v in current_range.values():
      if isinstance(v, tuple) and len(v) == 2:
        max_range = max(max_range, abs(v[0]) + abs(v[1]))
  return torch.tensor([max_range])


def command_velocity_range(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityRangeStage],
) -> torch.Tensor:
  """Update a command term's velocity_range based on training step stages.

  Used for curriculum: the motion command adds random velocity to the root when
  resampling; this adjusts that range over training.

  Args:
    env: The RL environment instance.
    env_ids: Environment indices (unused, kept for API consistency).
    command_name: Name of the command term (e.g. "motion").
    velocity_stages: List of stages with 'step' and 'velocity_range'. Ordered by step.

  Returns:
    Tensor with current max velocity range magnitude for logging.
  """
  del env_ids  # Unused.
  from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg

  command_term_cfg = env.command_manager.get_term_cfg(command_name)
  if command_term_cfg is None:
    return torch.tensor([0.0])
  motion_cfg = cast(MotionCommandCfg, command_term_cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      motion_cfg.velocity_range = stage["velocity_range"]

  current_range = motion_cfg.velocity_range or {}
  max_range = 0.0
  for v in current_range.values():
    if isinstance(v, tuple) and len(v) == 2:
      max_range = max(max_range, abs(v[0]) + abs(v[1]))
  return torch.tensor([max_range])


def termination_threshold(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  term_name: str,
  threshold_stages: list[ThresholdStage],
) -> torch.Tensor:
  """Update a termination term's threshold based on training step stages.
  
  This function allows you to adjust termination thresholds dynamically during training.
  Useful for curriculum learning where early training uses stricter thresholds and
  later training relaxes them (or vice versa).
  
  Args:
    env: The RL environment instance.
    env_ids: Environment indices (unused, kept for API consistency).
    term_name: Name of the termination term to adjust (must exist in terminations dict).
    threshold_stages: List of stages, each with 'step' (environment step) and 
                     'threshold' (target value). Stages should be ordered by step.
  
  Returns:
    Tensor containing the current threshold value for logging.
  
  Example:
    threshold_stages = [
      {"step": 0, "threshold": 0.15},            # Early: strict threshold
      {"step": 10000 * 24, "threshold": 0.20},   # Mid: relax slightly
      {"step": 20000 * 24, "threshold": 0.25},   # Late: final threshold
    ]
  """
  del env_ids  # Unused.
  term_cfg = env.termination_manager.get_term_cfg(term_name)
  for stage in threshold_stages:
    if env.common_step_counter > stage["step"]:
      term_cfg.params["threshold"] = stage["threshold"]
  return torch.tensor([term_cfg.params.get("threshold", 0.0)])
