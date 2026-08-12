from importlib.util import find_spec

from mjlab.tasks.fall.rl import FallOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

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

if find_spec("amp_rsl_rl") is not None:
  import rsl_rl.utils as _rsl_utils

  # amp-rsl-rl expects this helper, but newer rsl-rl releases may omit it.
  # Install the compatibility shim before importing the AMP runner.
  if not hasattr(_rsl_utils, "store_code_state"):

    def _store_code_state_noop(*_args: object, **_kwargs: object) -> list[object]:
      return []

    _rsl_utils.store_code_state = _store_code_state_noop

  from mjlab.rl.mj_amp_runner import MjlabAmpOnPolicyRunner

  register_mjlab_task(
    task_id="Mjlab-Falling-Flat-PM1-AMP",
    env_cfg=pm1_flat_falling_env_cfg(enable_amp_env=True),
    play_env_cfg=pm1_flat_falling_env_cfg(play=True, enable_amp_env=True),
    rl_cfg=pm1_falling_amp_runner_cfg(),
    runner_cls=MjlabAmpOnPolicyRunner,
  )
