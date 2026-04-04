"""RL configuration for PM1 tracking task."""

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlAmpAlgorithmCfg,
  RslRlPpoAlgorithmCfg,
)


def pm1_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for PM1 tracking task."""
  return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlAmpAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      # AMP reward mixing (actual per-env gating is implemented in
      # `mjlab/rl/mj_amp_runner.py` using `extras["recovery_mask"]`).
      task_reward_weight=2.0,
      disc_reward_weight=1.0,
      disc_reward_scale=1.0,
      disc_epochs=1,
      disc_batch_size_scale=2.0 / 24.0,
      disc_replay_samples=1000,
      disc_replay_buffer_size=200000,
      disc_lr=1.0e-4,
      disc_grad_penalty=10.0,
      disc_logit_reg=0.01,
      disc_input_noise_std=0.05,
      disc_hidden_dims=(512, 512),
    ),
    experiment_name="pm1_tracking_recovery_amp",
    save_interval=500,
    num_steps_per_env=32,
    max_iterations=30_000,
  )


def pm1_tracking_ppo_runner_no_amp_cfg() -> RslRlOnPolicyRunnerCfg:
  """Plain PPO (no AMP) for PM1 tracking + recovery."""
  return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="pm1_tracking_recovery_no_amp",
    save_interval=500,
    num_steps_per_env=30,
    max_iterations=30_000,
  )
