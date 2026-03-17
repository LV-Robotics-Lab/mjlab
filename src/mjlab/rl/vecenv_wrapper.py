from typing import Any, cast

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.utils.spaces import Space


def _make_amp_cfg_proxy(real_cfg: ManagerBasedRlEnvCfg, joint_names: list[str]) -> Any:
  """Build a proxy for env.cfg so amp-rsl-rl can access observations.amp and sim.dt."""
  asset_cfg = type("AssetCfg", (), {"joint_names": joint_names})()
  joint_pos = type("JointPos", (), {"params": {"asset_cfg": asset_cfg}})()
  amp = type("Amp", (), {"joint_pos": joint_pos})()
  obs = type("Observations", (), {"amp": amp})()

  class _SimProxy:
    def __init__(self, sim: Any) -> None:
      self._sim = sim

    def __getattr__(self, name: str) -> Any:
      # amp-rsl-rl expects cfg.sim.dt to be the physics timestep.
      if name == "dt":
        return self._sim.mujoco.timestep
      return getattr(self._sim, name)

  class _CfgProxy:
    def __getattr__(self, name: str) -> Any:
      if name == "observations":
        return obs
      if name == "sim":
        return _SimProxy(real_cfg.sim)
      return getattr(real_cfg, name)

  proxy = _CfgProxy()
  setattr(proxy, "_real_cfg", real_cfg)  # for wandb log_config: asdict(env_cfg) needs the real dataclass
  return proxy


class RslRlVecEnvWrapper(VecEnv):
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    clip_actions: float | None = None,
  ):
    self.env = env
    self.clip_actions = clip_actions

    self.num_envs = self.unwrapped.num_envs
    self.device = torch.device(self.unwrapped.device)
    self.max_episode_length = self.unwrapped.max_episode_length
    self.num_actions = self.unwrapped.action_manager.total_action_dim
    self._modify_action_space()

    # Reset at the start since rsl_rl does not call reset.
    self.env.reset()

  @property
  def cfg(self) -> ManagerBasedRlEnvCfg | Any:
    real = self.unwrapped.cfg
    amp_cfg = getattr(real, "amp", None)
    if amp_cfg is None:
      return real
    # amp-rsl-rl runner expects cfg.observations.amp.joint_pos.params["asset_cfg"].joint_names (list).
    robot = self.unwrapped.scene[amp_cfg.asset_name]
    joint_names = list(robot.joint_names)
    return _make_amp_cfg_proxy(real, joint_names)

  @property
  def render_mode(self) -> str | None:
    return self.env.render_mode

  @property
  def observation_space(self) -> Space:
    return self.env.observation_space

  @property
  def action_space(self) -> Space:
    return self.env.action_space

  @classmethod
  def class_name(cls) -> str:
    return cls.__name__

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self.env

  # Properties.

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return self.unwrapped.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:  # type: ignore
    self.unwrapped.episode_length_buf = value

  def seed(self, seed: int = -1) -> int:
    return self.unwrapped.seed(seed)

  def get_observations(self) -> TensorDict:
    obs_dict = dict(self.unwrapped.observation_manager.compute())
    amp_helper = getattr(self.unwrapped, "_amp_helper", None)
    if amp_helper is not None:
      obs_dict["amp"] = amp_helper.get_disc_obs()
    return TensorDict(cast(dict[str, Any], obs_dict), batch_size=[self.num_envs])

  def reset(self) -> tuple[TensorDict, dict]:
    obs_dict, extras = self.env.reset()
    amp_helper = getattr(self.unwrapped, "_amp_helper", None)
    if amp_helper is not None:
      obs_dict["amp"] = amp_helper.get_disc_obs()
    if "disc_obs" in extras:
      extras["amp_obs"] = extras["disc_obs"]
    return TensorDict(
      cast(dict[str, Any], obs_dict), batch_size=[self.num_envs]
    ), extras

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
    obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
    # Inject AMP observation into obs_dict for amp-rsl-rl runner.
    amp_helper = getattr(self.unwrapped, "_amp_helper", None)
    if amp_helper is not None:
      obs_dict["amp"] = amp_helper.get_disc_obs()
    term_or_trunc = terminated | truncated
    assert isinstance(rew, torch.Tensor)
    assert isinstance(term_or_trunc, torch.Tensor)
    dones = term_or_trunc.to(dtype=torch.long)
    if not self.cfg.is_finite_horizon:
      extras["time_outs"] = truncated
    # amp-rsl-rl 可能从 info/extras 读 disc 观测，兼容两种 key
    if "disc_obs" in extras:
      extras["amp_obs"] = extras["disc_obs"]
    return (
      TensorDict(cast(dict[str, Any], obs_dict), batch_size=[self.num_envs]),
      rew,
      dones,
      extras,
    )

  def close(self) -> None:
    return self.env.close()

  # AMP (amp-rsl-rl): delegate to unwrapped env when it provides disc_obs API.
  def get_disc_obs_space(self):
    """Return Box space for disc_obs. Only available when env has AMP configured."""
    if hasattr(self.unwrapped, "get_disc_obs_space"):
      return self.unwrapped.get_disc_obs_space()
    raise RuntimeError("AMP is not configured (env has no get_disc_obs_space).")

  def fetch_disc_obs_demo(self, num_samples: int) -> torch.Tensor:
    """Sample reference disc_obs for discriminator. Only when env has AMP configured."""
    if hasattr(self.unwrapped, "fetch_disc_obs_demo"):
      return self.unwrapped.fetch_disc_obs_demo(num_samples)
    raise RuntimeError("AMP is not configured (env has no fetch_disc_obs_demo).")

  # Private methods.

  def _modify_action_space(self) -> None:
    if self.clip_actions is None:
      return

    from mjlab.utils.spaces import Box, batch_space

    self.unwrapped.single_action_space = Box(
      shape=(self.num_actions,), low=-self.clip_actions, high=self.clip_actions
    )
    self.unwrapped.action_space = batch_space(
      self.unwrapped.single_action_space, self.num_envs
    )
