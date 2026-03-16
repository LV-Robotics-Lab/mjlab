from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.fall.rl import FallOnPolicyRunner

from .env_cfgs import pm1_flat_falling_env_cfg
from .rl_cfg import pm1_falling_amp_runner_cfg, pm1_falling_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Falling-Flat-PM1",
  env_cfg=pm1_flat_falling_env_cfg(),
  play_env_cfg=pm1_flat_falling_env_cfg(play=True),
  rl_cfg=pm1_falling_ppo_runner_cfg(),
  runner_cls=FallOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Falling-Flat-PM1-No-State-Estimation",
  env_cfg=pm1_flat_falling_env_cfg(has_state_estimation=False),
  play_env_cfg=pm1_flat_falling_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=pm1_falling_ppo_runner_cfg(),
  runner_cls=FallOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Falling-Flat-PM1-AMP",
  env_cfg=pm1_flat_falling_env_cfg(),
  play_env_cfg=pm1_flat_falling_env_cfg(play=True),
  rl_cfg=pm1_falling_amp_runner_cfg(),
  runner_cls=FallOnPolicyRunner,
)

