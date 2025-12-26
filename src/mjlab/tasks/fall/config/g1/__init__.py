from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.fall.rl import FallOnPolicyRunner

from .env_cfgs import unitree_g1_flat_fall_env_cfg
from .rl_cfg import unitree_g1_fall_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Fall-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_fall_env_cfg(),
  play_env_cfg=unitree_g1_flat_fall_env_cfg(play=True),
  rl_cfg=unitree_g1_fall_ppo_runner_cfg(),
  runner_cls=FallOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Fall-Flat-Unitree-G1-No-State-Estimation",
  env_cfg=unitree_g1_flat_fall_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_fall_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_g1_fall_ppo_runner_cfg(),
  runner_cls=FallOnPolicyRunner,
)
