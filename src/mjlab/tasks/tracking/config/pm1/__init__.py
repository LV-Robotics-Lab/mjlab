from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import (
  MotionTrackingDaggerRunner,
  MotionTrackingOnPolicyRunner,
)

from .env_cfgs import pm1_distill_env_cfg, pm1_flat_tracking_env_cfg
from .rl_cfg import pm1_fall_protection_dagger_runner_cfg, pm1_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-PM1",
  env_cfg=pm1_flat_tracking_env_cfg(),
  play_env_cfg=pm1_flat_tracking_env_cfg(play=True),
  rl_cfg=pm1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-PM1-No-State-Estimation",
  env_cfg=pm1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=pm1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=pm1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Fall-Protection-PM1-Distill",
  env_cfg=pm1_distill_env_cfg(),
  play_env_cfg=pm1_distill_env_cfg(play=True),
  rl_cfg=pm1_fall_protection_dagger_runner_cfg(),
  runner_cls=MotionTrackingDaggerRunner,
)
