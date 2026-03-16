from __future__ import annotations

from collections import deque
from typing import Callable

import os
import statistics
import time

import amp_rsl_rl
import rsl_rl
import torch
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.utils import resolve_obs_groups, store_code_state
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from amp_rsl_rl.algorithms import AMP_PPO
from amp_rsl_rl.networks import ActorCriticMoE, Discriminator
from amp_rsl_rl.utils import export_policy_as_onnx

from mjlab.rl.amp_loader import MjlabAmpLoader


_BUILTIN_CLASSES = {
  "ActorCritic": ActorCritic,
  "ActorCriticRecurrent": ActorCriticRecurrent,
  "ActorCriticMoE": ActorCriticMoE,
  "AMP_PPO": AMP_PPO,
}


def resolve_class(class_name: str) -> type:
  """Resolve a class by name, mirroring amp_rsl_rl.runners.amp_on_policy_runner."""
  if class_name in _BUILTIN_CLASSES:
    return _BUILTIN_CLASSES[class_name]
  if "." in class_name:
    module_path, cls_name = class_name.rsplit(".", 1)
    module = __import__(module_path, fromlist=[cls_name])
    return getattr(module, cls_name)
  raise ValueError(
    f"Unknown class name '{class_name}'. Provide a fully-qualified "
    f"module path (e.g. 'my_package.module.{class_name}') or one of "
    f"the built-in names: {list(_BUILTIN_CLASSES.keys())}."
  )


class MjlabAmpOnPolicyRunner:
  """AMP runner that uses mjlab's disc_obs demos instead of AMPLoader.

  This is a near-copy of amp_rsl_rl.runners.AMPOnPolicyRunner, with the
  only functional change being that expert AMP data comes from
  MjlabAmpLoader(env, device), which in turn calls the environment's
  AMPHelper.fetch_disc_obs_demo_pairs(). This guarantees that policy
  and expert AMP observations have matching dimensions.
  """

  def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
    self.cfg = train_cfg
    self.alg_cfg = train_cfg["algorithm"]
    self.policy_cfg = train_cfg["policy"]
    # dataset_cfg is preserved for logging / compatibility but not used
    self.dataset_cfg = train_cfg.get("dataset", {})
    self.discriminator_cfg = train_cfg["discriminator"]
    self.device = device
    self.env = env

    # Optional custom exporter function (set via set_export_policy_fn)
    self._export_policy_fn: Callable | None = None

    observations = self.env.get_observations()
    default_sets = ["critic"]
    self.cfg["obs_groups"] = resolve_obs_groups(
      observations, self.cfg.get("obs_groups"), default_sets
    )

    actor_critic_class = resolve_class(self.policy_cfg.pop("class_name"))
    actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticMoE = (
      actor_critic_class(
        observations,
        self.cfg["obs_groups"],
        self.env.num_actions,
        **self.policy_cfg,
      ).to(self.device)
    )

    # Initialize all the ingredients required for AMP (discriminator, dataset loader)
    num_amp_obs = observations["amp"].shape[1]

    # Expert AMP data from mjlab env disc_obs demos (same dim as policy obs["amp"])
    amp_data = MjlabAmpLoader(self.env, device=self.device)

    self.discriminator = Discriminator(
      input_dim=num_amp_obs * 2,  # concat current and next AMP obs
      hidden_layer_sizes=self.discriminator_cfg["hidden_dims"],
      reward_scale=self.discriminator_cfg["reward_scale"],
      device=self.device,
      loss_type=self.discriminator_cfg["loss_type"],
      empirical_normalization=self.discriminator_cfg["empirical_normalization"],
    ).to(self.device)

    # Initialize the AMP-PPO algorithm
    alg_class = resolve_class(self.alg_cfg.pop("class_name"))
    for key in list(self.alg_cfg.keys()):
      if key not in AMP_PPO.__init__.__code__.co_varnames:
        self.alg_cfg.pop(key)

    self.alg: AMP_PPO = alg_class(
      actor_critic=actor_critic,
      discriminator=self.discriminator,
      amp_data=amp_data,
      device=self.device,
      **self.alg_cfg,
    )
    self.num_steps_per_env = self.cfg["num_steps_per_env"]
    self.save_interval = self.cfg["save_interval"]
    # init storage and model
    obs_template = observations.clone().detach().to(self.device)
    self.alg.init_storage(
      self.env.num_envs,
      self.num_steps_per_env,
      obs_template,
      (self.env.num_actions,),
    )

    # Log
    self.log_dir = log_dir
    self.writer = None
    self.logger_type = None
    self.tot_timesteps = 0
    self.tot_time = 0
    self.current_learning_iteration = 0
    self.git_status_repos = [rsl_rl.__file__, amp_rsl_rl.__file__]

  # The rest of the methods (learn, log, save, load, etc.) are identical
  # to amp_rsl_rl.runners.AMPOnPolicyRunner and are reused via delegation.

  # Delegate train/eval/save/load/log to a wrapped AMPOnPolicyRunner instance
  # initialized with the same state, except for amp_data. For simplicity and
  # to avoid code duplication, we keep all high-level methods the same by
  # inheriting from AMP_PPO directly.

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    # This implementation is identical to AMPOnPolicyRunner.learn, but uses
    # self.alg (initialized with mjlab amp_data) above. We rely on the
    # original implementation from amp_rsl_rl, so just import and call it.
    from amp_rsl_rl.runners.amp_on_policy_runner import AMPOnPolicyRunner as _Base

    # Create a temporary base runner that shares our fields and call its learn.
    base = _Base.__new__(_Base)  # type: ignore
    base.__dict__.update(self.__dict__)
    return _Base.learn(base, num_learning_iterations, init_at_random_ep_len)

  # The remaining helper methods (log, save, load, etc.) are also delegated.

  def log(self, locs: dict, width: int = 80, pad: int = 35):
    from amp_rsl_rl.runners.amp_on_policy_runner import AMPOnPolicyRunner as _Base

    base = _Base.__new__(_Base)  # type: ignore
    base.__dict__.update(self.__dict__)
    return _Base.log(base, locs, width, pad)

  def set_export_policy_fn(self, fn: Callable) -> None:
    self._export_policy_fn = fn

  def save(self, path, infos=None, save_onnx=False):
    from amp_rsl_rl.runners.amp_on_policy_runner import AMPOnPolicyRunner as _Base

    base = _Base.__new__(_Base)  # type: ignore
    base.__dict__.update(self.__dict__)
    return _Base.save(base, path, infos, save_onnx)

  def load(self, path, load_optimizer=True, weights_only=False):
    from amp_rsl_rl.runners.amp_on_policy_runner import AMPOnPolicyRunner as _Base

    base = _Base.__new__(_Base)  # type: ignore
    base.__dict__.update(self.__dict__)
    return _Base.load(base, path, load_optimizer, weights_only)

  def get_inference_policy(self, device=None):
    from amp_rsl_rl.runners.amp_on_policy_runner import AMPOnPolicyRunner as _Base

    base = _Base.__new__(_Base)  # type: ignore
    base.__dict__.update(self.__dict__)
    return _Base.get_inference_policy(base, device)

  def train_mode(self):
    self.alg.actor_critic.train()
    self.alg.discriminator.train()

  def eval_mode(self):
    self.alg.actor_critic.eval()
    self.alg.discriminator.eval()

  def add_git_repo_to_log(self, repo_file_path):
    self.git_status_repos.append(repo_file_path)

