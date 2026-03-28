from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl.mj_amp_runner import MjlabAmpOnPolicyRunner
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import pm1_flat_tracking_env_cfg
from .rl_cfg import pm1_tracking_ppo_runner_cfg, pm1_tracking_ppo_runner_no_amp_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-PM1",
  env_cfg=pm1_flat_tracking_env_cfg(),
  play_env_cfg=pm1_flat_tracking_env_cfg(play=True),
  rl_cfg=pm1_tracking_ppo_runner_cfg(),
  runner_cls=MjlabAmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-PM1-No-State-Estimation",
  env_cfg=pm1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=pm1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=pm1_tracking_ppo_runner_cfg(),
  runner_cls=MjlabAmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-PM1-No-AMP",
  env_cfg=pm1_flat_tracking_env_cfg(use_amp=False),
  play_env_cfg=pm1_flat_tracking_env_cfg(use_amp=False, play=True),
  rl_cfg=pm1_tracking_ppo_runner_no_amp_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
