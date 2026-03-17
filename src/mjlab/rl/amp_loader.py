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

    Demo windows are precomputed once inside AMPHelper, so each mini-batch can
    now be sampled cheaply with random indexing.
    """
    helper = getattr(self.env.unwrapped, "_amp_helper", None)
    if helper is None:
      raise RuntimeError("AMP is not configured (env has no _amp_helper).")

    for _ in range(num_mini_batch):
      states, next_states = helper.fetch_disc_obs_demo_pairs(mini_batch_size)
      yield states.to(self.device), next_states.to(self.device)

