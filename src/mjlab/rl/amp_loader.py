from __future__ import annotations

from typing import Generator, Tuple

import torch

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class MjlabAmpLoader:
  """Expert AMP data loader that uses mjlab's disc_obs demos.

  This replaces amp_rsl_rl.utils.AMPLoader in our setup. It samples
  (state, next_state) pairs directly from the environment's AMPHelper,
  so that expert AMP observations have exactly the same dimension as
  the policy's obs['amp'] (including history, root pose, etc.).
  """

  def __init__(self, env: RslRlVecEnvWrapper, device: str = "cpu") -> None:
    self.env = env
    self.device = torch.device(device)

  def feed_forward_generator(
    self,
    num_mini_batch: int,
    mini_batch_size: int,
  ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """Yield mini-batches of (state, next_state) demo AMP observations.

    Shapes:
      state:      (mini_batch_size, disc_dim)
      next_state: (mini_batch_size, disc_dim)

    Fetches one large batch per learning iteration and slices into mini-batches
    to avoid repeated Python-side sampling and compute_disc_obs calls.
    """
    helper = getattr(self.env.unwrapped, "_amp_helper", None)
    if helper is None:
      raise RuntimeError("AMP is not configured (env has no _amp_helper).")

    total = num_mini_batch * mini_batch_size
    states, next_states = helper.fetch_disc_obs_demo_pairs(total)
    states = states.to(self.device)
    next_states = next_states.to(self.device)
    for i in range(num_mini_batch):
      start = i * mini_batch_size
      end = start + mini_batch_size
      yield states[start:end], next_states[start:end]

