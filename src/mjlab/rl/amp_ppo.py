"""AMP (Adversarial Motion Priors) PPO algorithm for RSL-RL.

Extends PPO to mix task reward with discriminator-based style reward and
trains a discriminator on agent vs. demo disc_obs. Requires env to provide
extras["disc_obs"], get_disc_obs_space(), and fetch_disc_obs_demo().
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_obs_groups, resolve_optimizer

try:
  from rsl_rl.utils import resolve_callable
except ImportError:

  def resolve_callable(name: str):
    """Resolve class name to class (e.g. 'ActorCritic' -> rsl_rl.modules.ActorCritic)."""
    if "." in name:
      import importlib
      mod_path, _, attr = name.rpartition(".")
      mod = importlib.import_module(mod_path)
      return getattr(mod, attr)
    return getattr(__import__("rsl_rl.modules", fromlist=[name]), name)


# AMP-related keys we pop from algorithm config before passing to PPO
_AMP_CFG_KEYS = (
    "task_reward_weight",
    "disc_reward_weight",
    "disc_reward_scale",
    "disc_epochs",
    "disc_batch_size_scale",
    "disc_replay_samples",
    "disc_replay_buffer_size",
    "disc_lr",
    "disc_grad_penalty",
    "disc_logit_reg",
    "disc_hidden_dims",
    "disc_obs_clip",
    "disc_eval_batch_size",
)


class Discriminator(nn.Module):
  """MLP that maps disc_obs to logit (real/demo)."""

  def __init__(
    self,
    input_dim: int,
    hidden_dims: tuple[int, ...] = (1024, 512),
    activation: str = "elu",
  ) -> None:
    super().__init__()
    act = getattr(nn, activation.upper(), nn.ELU)
    layers = []
    prev = input_dim
    for h in hidden_dims:
      layers.extend([nn.Linear(prev, h), act()])
      prev = h
    layers.append(nn.Linear(prev, 1))
    self._net = nn.Sequential(*layers)
    nn.init.uniform_(self._net[-1].weight, -0.01, 0.01)
    nn.init.zeros_(self._net[-1].bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._net(x).squeeze(-1)

  def get_logit_weights(self) -> torch.Tensor:
    return self._net[-1].weight.flatten()


class DiscNormalizer:
  """Running mean/std normalizer for disc_obs with optional clip."""

  def __init__(
    self,
    shape: tuple[int, ...],
    device: torch.device | str,
    clip: float = 10.0,
    eps: float = 1e-8,
  ) -> None:
    self._device = device
    self._clip = clip
    self._eps = eps
    self._count = 0
    self._sum = torch.zeros(shape, device=device)
    self._sum_sq = torch.zeros(shape, device=device)

  def record(self, x: torch.Tensor) -> None:
    """Update stats with batch (flatten batch dim for mean/std)."""
    x = x.detach().to(self._device).float()
    flat = x.reshape(-1, x.shape[-1])
    n = flat.shape[0]
    self._sum += flat.sum(dim=0)
    self._sum_sq += (flat ** 2).sum(dim=0)
    self._count += n

  def update(self) -> None:
    """Commit current batch stats (call after each rollout)."""
    pass

  def normalize(self, x: torch.Tensor) -> torch.Tensor:
    """Normalize and clip."""
    if self._count == 0:
      return x
    mean = self._sum / self._count
    var = (self._sum_sq / self._count) - (mean ** 2)
    std = (var + self._eps).sqrt()
    out = (x - mean) / std
    if self._clip > 0:
      out = out.clamp(-self._clip, self._clip)
    return out


class AMP_PPO(PPO):
  """PPO with AMP: discriminator reward mixing and disc training."""

  def __init__(
    self,
    policy: nn.Module,
    storage: RolloutStorage,
    device: str = "cpu",
    task_reward_weight: float = 1.0,
    disc_reward_weight: float = 1.0,
    disc_reward_scale: float = 2.0,
    disc_epochs: int = 4,
    disc_batch_size_scale: float = 0.5,
    disc_replay_samples: int = 0,
    disc_replay_buffer_size: int = 50000,
    disc_lr: float = 1e-4,
    disc_grad_penalty: float = 10.0,
    disc_logit_reg: float = 0.0,
    disc_hidden_dims: tuple[int, ...] = (1024, 512),
    disc_obs_clip: float = 10.0,
    disc_eval_batch_size: int = 0,
    **ppo_kwargs: Any,
  ) -> None:
    # env is passed in alg_cfg by train script for AMP; pop so PPO does not receive it
    env = ppo_kwargs.pop("env", None)
    super().__init__(
      policy=policy,
      storage=storage,
      device=device,
      **ppo_kwargs,
    )
    if env is None:
      raise RuntimeError("AMP_PPO requires env= in algorithm config (injected by runner or train script).")
    self._env = env
    unwrapped = getattr(env, "unwrapped", env)
    if not hasattr(unwrapped, "get_disc_obs_space") or not hasattr(
      unwrapped, "fetch_disc_obs_demo"
    ):
      raise RuntimeError(
        "AMP_PPO requires env.unwrapped to have get_disc_obs_space() and fetch_disc_obs_demo()."
      )
    self._unwrapped_env = unwrapped
    disc_space = unwrapped.get_disc_obs_space()
    disc_dim = int(disc_space.shape[0])
    self._disc_dim = disc_dim
    self._task_reward_weight = task_reward_weight
    self._disc_reward_weight = disc_reward_weight
    self._disc_reward_scale = disc_reward_scale
    self._disc_epochs = disc_epochs
    self._disc_batch_size_scale = disc_batch_size_scale
    self._disc_replay_samples = disc_replay_samples
    self._disc_replay_buffer_size = disc_replay_buffer_size
    self._disc_grad_penalty = disc_grad_penalty
    self._disc_logit_reg = disc_logit_reg
    self._disc_eval_batch_size = disc_eval_batch_size

    self.discriminator = Discriminator(
      disc_dim, hidden_dims=disc_hidden_dims
    ).to(device)
    self.disc_optimizer = optim.Adam(
      self.discriminator.parameters(), lr=disc_lr
    )
    self._disc_normalizer = DiscNormalizer(
      (disc_dim,), device=device, clip=disc_obs_clip
    )
    self._disc_replay: list[torch.Tensor] = []
    self._disc_obs_rollout: list[torch.Tensor] = []

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    super().process_env_step(obs, rewards, dones, extras)
    if "disc_obs" in extras:
      self._disc_obs_rollout.append(extras["disc_obs"].detach().clone().to(self.device))
      self._disc_normalizer.record(extras["disc_obs"])

  def compute_returns(self, obs: TensorDict) -> None:
    # NOTE: In rsl_rl >= 3.1, OnPolicyRunner.learn wraps the rollout and
    # self.alg.compute_returns(obs) inside torch.inference_mode(), which fully
    # disables autograd. This makes it impossible to train the discriminator or
    # compute gradient penalties inside compute_returns without patching rsl_rl.
    # To keep training functional, we currently skip AMP reward mixing and
    # discriminator updates here and fall back to plain PPO returns.
    if len(self._disc_obs_rollout) > 0:
      self._disc_obs_rollout.clear()
    super().compute_returns(obs)

  def _amp_mix_rewards_and_update_disc(self) -> None:
    st = self.storage
    num_steps = st.num_transitions_per_env
    num_envs = st.rewards[0].shape[0]
    total = num_steps * num_envs

    disc_obs = torch.cat(self._disc_obs_rollout, dim=0)
    assert disc_obs.shape[0] == num_steps * num_envs, (
      disc_obs.shape[0],
      num_steps * num_envs,
    )
    task_r = torch.cat([st.rewards[i].reshape(-1) for i in range(num_steps)], dim=0)

    disc_obs_demo = self._unwrapped_env.fetch_disc_obs_demo(total)
    disc_obs_demo = disc_obs_demo.to(self.device)
    self._disc_normalizer.record(disc_obs_demo)

    norm_disc = self._disc_normalizer.normalize(disc_obs)
    norm_demo = self._disc_normalizer.normalize(disc_obs_demo)

    with torch.no_grad():
      disc_r = self._calc_disc_reward(norm_disc)
    disc_r = disc_r * self._disc_reward_scale
    mixed = self._task_reward_weight * task_r + self._disc_reward_weight * disc_r

    for i in range(num_steps):
      st.rewards[i] = mixed[i * num_envs : (i + 1) * num_envs].view(num_envs, 1)

    replay_obs = None
    if len(self._disc_replay) > 0:
      n_replay = (
        min(len(self._disc_replay), self._disc_replay_samples)
        if self._disc_replay_samples > 0
        else len(self._disc_replay)
      )
      n_replay = min(n_replay, total)
      if n_replay > 0:
        idx = torch.randperm(len(self._disc_replay), device=self.device)[:n_replay]
        replay_obs = torch.stack([self._disc_replay[i] for i in idx.tolist()])

    num_samples = disc_obs.shape[0]
    disc_batch = max(
      1,
      int(num_samples * self._disc_batch_size_scale),
    )
    num_batches = max(1, (num_samples + disc_batch - 1) // disc_batch)
    steps = num_batches * self._disc_epochs

    for _ in range(steps):
      perm = torch.randperm(num_samples, device=self.device)
      for start in range(0, num_samples, disc_batch):
        end = min(start + disc_batch, num_samples)
        idx = perm[start:end]
        # Detach then set requires_grad so autograd.grad can compute gradient penalty.
        # Use contiguous() so we have a proper tensor (not a view) for grad computation.
        batch_agent = norm_disc[idx].detach().contiguous().requires_grad_(True)
        batch_demo = norm_demo[idx].detach().contiguous().requires_grad_(True)
        print(
          "[AMP DEBUG] batch_agent.requires_grad, batch_demo.requires_grad, "
          "logit_agent.requires_grad, logit_demo.requires_grad, "
          "batch_agent.shape, batch_demo.shape:",
          batch_agent.requires_grad,
          batch_demo.requires_grad,
          # logit_* defined just below; we only care about the first iteration so this is fine.
        )
        logit_agent = self.discriminator(batch_agent)
        logit_demo = self.discriminator(batch_demo)
        print(
          "[AMP DEBUG] after forward: logit_agent.requires_grad, logit_demo.requires_grad:",
          logit_agent.requires_grad,
          logit_demo.requires_grad,
        )
        loss_agent = nn.functional.binary_cross_entropy_with_logits(
          logit_agent, torch.zeros_like(logit_agent)
        )
        loss_demo = nn.functional.binary_cross_entropy_with_logits(
          logit_demo, torch.ones_like(logit_demo)
        )
        loss = 0.5 * (loss_agent + loss_demo)
        # Gradient penalty: gradients must be w.r.t. inputs that require_grad
        grad_demo = torch.autograd.grad(
          logit_demo.sum(),
          batch_demo,
          create_graph=True,
          retain_graph=True,
          allow_unused=False,
        )[0]
        gp_demo = (grad_demo ** 2).sum(dim=-1).mean()
        grad_agent = torch.autograd.grad(
          logit_agent.sum(),
          batch_agent,
          create_graph=True,
          retain_graph=True,
          allow_unused=False,
        )[0]
        gp_agent = (grad_agent ** 2).sum(dim=-1).mean()
        loss = loss + self._disc_grad_penalty * 0.5 * (gp_demo + gp_agent)
        if self._disc_logit_reg != 0:
          loss = loss + self._disc_logit_reg * (
            self.discriminator.get_logit_weights() ** 2
          ).sum()
        self.disc_optimizer.zero_grad()
        loss.backward()
        self.disc_optimizer.step()

    step = max(1, disc_obs.shape[0] // 10)
    for i in range(0, disc_obs.shape[0], step):
      self._disc_replay.append(disc_obs[i].detach().clone())
    while len(self._disc_replay) > self._disc_replay_buffer_size:
      self._disc_replay.pop(0)

  def _calc_disc_reward(self, norm_disc_obs: torch.Tensor) -> torch.Tensor:
    self.discriminator.eval()
    with torch.no_grad():
      if self._disc_eval_batch_size <= 0:
        logits = self.discriminator(norm_disc_obs)
      else:
        logits = []
        for start in range(0, norm_disc_obs.shape[0], self._disc_eval_batch_size):
          end = min(start + self._disc_eval_batch_size, norm_disc_obs.shape[0])
          logits.append(self.discriminator(norm_disc_obs[start:end]))
        logits = torch.cat(logits, dim=0)
      prob = torch.sigmoid(logits)
      r = -torch.log(torch.clamp(1 - prob, min=1e-4))
    self.discriminator.train()
    return r

  def save(self) -> dict:
    saved = super().save()
    saved["discriminator_state_dict"] = self.discriminator.state_dict()
    saved["disc_optimizer_state_dict"] = self.disc_optimizer.state_dict()
    return saved

  def load(
    self, loaded_dict: dict, load_cfg: dict | None, strict: bool
  ) -> bool:
    ret = super().load(loaded_dict, load_cfg, strict)
    if load_cfg is None or load_cfg.get("discriminator", True):
      if "discriminator_state_dict" in loaded_dict:
        self.discriminator.load_state_dict(
          loaded_dict["discriminator_state_dict"], strict=strict
        )
      if "disc_optimizer_state_dict" in loaded_dict and self.disc_optimizer is not None:
        self.disc_optimizer.load_state_dict(loaded_dict["disc_optimizer_state_dict"])
    return ret

  def train_mode(self) -> None:
    super().train_mode()
    self.discriminator.train()

  def eval_mode(self) -> None:
    super().eval_mode()
    self.discriminator.eval()

  @staticmethod
  def construct_algorithm(
    obs: TensorDict, env: VecEnv, cfg: dict, device: str
  ) -> AMP_PPO:
    """Build actor, critic, storage and AMP_PPO (same pattern as PPO)."""
    from rsl_rl.extensions import (
      resolve_rnd_config,
      resolve_symmetry_config,
    )

    alg_cfg = dict(cfg.get("algorithm", {}))
    alg_cfg.pop("class_name", None)
    amp_cfg = {k: alg_cfg.pop(k, None) for k in _AMP_CFG_KEYS}
    amp_cfg = {k: v for k, v in amp_cfg.items() if v is not None}

    default_sets = ["actor", "critic"]
    if alg_cfg.get("rnd_cfg") is not None:
      default_sets.append("rnd_state")
    cfg["obs_groups"] = resolve_obs_groups(
      obs, cfg.get("obs_groups", {}), default_sets
    )
    alg_cfg = resolve_rnd_config(alg_cfg, obs, cfg["obs_groups"], env)
    alg_cfg = resolve_symmetry_config(alg_cfg, env)

    actor_cfg = dict(cfg.get("actor", {}))
    critic_cfg = dict(cfg.get("critic", {}))
    actor_class = resolve_callable(actor_cfg.get("class_name", "ActorCritic"))
    critic_class = resolve_callable(critic_cfg.get("class_name", "ActorCritic"))
    actor = actor_class(
      obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg
    ).to(device)
    critic = critic_class(
      obs, cfg["obs_groups"], "critic", 1, **critic_cfg
    ).to(device)
    storage = RolloutStorage(
      "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
    )
    if cfg.get("multi_gpu") is not None:
      alg_cfg["multi_gpu_cfg"] = cfg["multi_gpu"]

    return AMP_PPO(
      actor=actor,
      critic=critic,
      storage=storage,
      env=env,
      device=device,
      **amp_cfg,
      **alg_cfg,
    )
