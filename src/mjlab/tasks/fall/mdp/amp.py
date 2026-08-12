"""AMP (Adversarial Motion Priors) for fall task.

Re-exports from mjlab.envs.amp. Use AMPCfg in fall env_cfg to enable disc_obs,
get_disc_obs_space(), and fetch_disc_obs_demo() for AMP-style training.
"""

from mjlab.envs.amp import AMPCfg, calc_disc_obs_dim, compute_disc_obs

__all__ = ["AMPCfg", "calc_disc_obs_dim", "compute_disc_obs"]
