"""RL configuration for PM1 tracking task."""

from mjlab.rl import (
  RslRlDaggerRunnerCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
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
    experiment_name="pm1_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
    clip_actions=1000.0,  # Match ROS2 action_clip
  )


def pm1_fall_protection_dagger_runner_cfg(
  teacher_forward_checkpoint: str = "",
  teacher_backward_checkpoint: str = "",
  dagger_coef: float = 0.15,
  dagger_coef_anneal_steps: int = 80_000,
  dagger_coef_min: float = 0.06,
) -> RslRlDaggerRunnerCfg:
  """双 Teacher 摔倒防护蒸馏：前摔/后摔各一个 Teacher，按 reset 初速度方向选择。"""
  return RslRlDaggerRunnerCfg(
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
      schedule="fixed",  # distill 时 KL(teacher||student) 会带来较大 policy 变化，adaptive 易把 LR 压到 1e-5 导致不学习
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="pm1_fall_protection_distill",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
    clip_actions=1000.0,
    teacher_forward_checkpoint=teacher_forward_checkpoint,
    teacher_backward_checkpoint=teacher_backward_checkpoint,
    dagger_coef=dagger_coef,
    dagger_coef_anneal_steps=dagger_coef_anneal_steps,
    dagger_coef_min=dagger_coef_min,
  )
