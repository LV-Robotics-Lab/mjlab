"""Fall task configuration.

This module provides a factory function to create a base fall task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
  CurriculumTermCfg,
  ActionTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.envs.amp import AMPCfg
from mjlab.tasks.fall import mdp
from mjlab.tasks.fall.mdp.curriculums import (
  reset_initialization_curriculum,
  reset_push_curriculum,
)
from mjlab.tasks.fall.mdp.events import push_by_setting_velocity_preserve_data
from mjlab.tasks.fall.mdp.rewards import base_height_reward, upright_reward
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_fall_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base fall task configuration."""

  ##
  # Observations
  ##

  # Actor (policy): deployable sensor measurements only.
  policy_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    # Projected gravity g_b in pelvis frame, encoding roll / pitch.
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    # Joint positions q relative to defaults.
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Joint velocities q̇ relative to defaults.
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    # Previous action a_{t-1}.
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  # Critic: same as actor plus privileged base (shorter history for speed).
  critic_terms = {
    **policy_terms,
    "base_pos": ObservationTermCfg(
      func=mdp.base_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot")},
      history_length=2,
      flatten_history_dim=True,
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      history_length=2,
      flatten_history_dim=True,
    ),
  }

  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      asset_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_mixed,
      mode="reset",
      params={
        "tilt_pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.02, 0.08),
          "roll": (-0.3, 0.3),
          "pitch": (-0.3, 0.3),
          "yaw": (-3.14, 3.14),
        },
        "tilt_velocity_range": {},
        "tilt_joint_position_range": (-0.25, 0.25),
        "tilt_joint_velocity_range": (-0.05, 0.05),
        "data_probability": 0.0,
        "motion_files": (),
        "data_root_body_name": "LINK_BASE",
        "data_pose_range": {
          "x": (-0.02, 0.02),
          "y": (-0.02, 0.02),
          "z": (-0.01, 0.01),
          "roll": (-0.05, 0.05),
          "pitch": (-0.05, 0.05),
          "yaw": (-0.10, 0.10),
        },
        "data_velocity_range": {
          "x": (-0.10, 0.10),
          "y": (-0.10, 0.10),
          "z": (-0.05, 0.05),
          "roll": (-0.10, 0.10),
          "pitch": (-0.10, 0.10),
          "yaw": (-0.10, 0.10),
        },
        "data_joint_position_range": (-0.03, 0.03),
        "data_joint_velocity_range": (-0.05, 0.05),
      },
    ),
    # Apply an extra reset push only to non-data initializations so motion-derived
    # root velocities remain unchanged.
    "push_at_reset": EventTermCfg(
      func=push_by_setting_velocity_preserve_data,
      mode="reset",
      params={
        "velocity_range": {
          "x": (-2.0, 2.0),
          "y": (-2.0, 2.0),
          "z": (-0.2, 0.2),
          "roll": (-0.4, 0.4),
          "pitch": (-0.4, 0.4),
          "yaw": (-0.5, 0.5),
        },
        "preserve_data_reset_states": True,
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 2.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.2, 0.2),
          "roll": (-0.3, 0.3),
          "pitch": (-0.3, 0.3),
          "yaw": (-0.4, 0.4),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "field": "geom_friction",
        "ranges": (0.3, 1.2),
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    # "upright": RewardTermCfg(
    #   func=upright_reward,
    #   weight=0.5,
    #   params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.4},
    # ),
    # "base_height": RewardTermCfg(
    #   func=base_height_reward,
    #   weight=0.3,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot"),
    #     "nominal_height": 0.0,
    #     "std": 0.1,
    #   },
    # ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision"},
    ),
    "reduce_contact_force": RewardTermCfg(
      func=mdp.reduce_contact_force_weighted,
      weight=0.01, # 0.01
      params={
        "sensor_name": "body_contact_force",
        "high_weight_bodies": (
          "LINK_ELBOW_END_L",
          "LINK_ELBOW_END_R",
          "LINK_HEAD_YAW",
          "LINK_TORSO_YAW",
        ),
        "medium_weight_bodies": (
          "LINK_ELBOW_PITCH_L",
          "LINK_ELBOW_PITCH_R",
          "LINK_ELBOW_YAW_L",
          "LINK_ELBOW_YAW_R",
        ),
        "shoulder_weight_bodies": (
          "LINK_SHOULDER_ROLL_L",
          "LINK_SHOULDER_ROLL_R",
          "LINK_SHOULDER_YAW_L",
          "LINK_SHOULDER_YAW_R",
        ),
        "high_weight": 10.0,
        "shoulder_weight": 5.0,
        "medium_weight": 2.0,
        "low_weight": 0.5,
        "alpha": 0.3,
      },
    ),
    "control_descent_speed": RewardTermCfg(
      func=mdp.control_descent_speed,
      weight=1,
      params={
        "torso_body_name": "LINK_TORSO_YAW",
        "threshold": 0.5,
      },
    ),
    # "impact_velocity_reward": RewardTermCfg(
    #   func=mdp.ImpactVelocityReward(
    #     sensor_name="body_contact_force",
    #     high_weight_bodies=(
    #       "LINK_ELBOW_END_L",
    #       "LINK_ELBOW_END_R",
    #       "LINK_HEAD_YAW",
    #       "LINK_TORSO_YAW",
    #     ),
    #     medium_weight_bodies=(
    #       "LINK_ELBOW_PITCH_L",
    #       "LINK_ELBOW_PITCH_R",
    #       "LINK_ELBOW_YAW_L",
    #       "LINK_ELBOW_YAW_R",
    #     ),
    #     shoulder_weight_bodies=(
    #       "LINK_SHOULDER_ROLL_L",
    #       "LINK_SHOULDER_ROLL_R",
    #       "LINK_SHOULDER_YAW_L",
    #       "LINK_SHOULDER_YAW_R",
    #     ),
    #     high_weight=10.0,
    #     shoulder_weight=5.0,
    #     medium_weight=2.0,
    #     low_weight=0.5,
    #   ),
    #   weight=1,
    # ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "forbidden_body_contact_force": TerminationTermCfg(
      func=mdp.bad_body_contact_force,
      params={
        "sensor_name": "body_contact_force",
        "body_names": (),  # Set per-robot.
        "force_threshold": 1e9,  # Set per-robot.
      },
    ),
    # "fell_over": TerminationTermCfg(
    #   func=mdp.bad_orientation,
    #   params={"limit_angle": math.radians(80.0)},
    # ),
    # "anchor_pos": TerminationTermCfg(
    #   func=mdp.bad_base_pos_z_only,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot"),
    #     "threshold": 0.35,
    #   },
    # ),
    # "ee_body_pos": TerminationTermCfg(
    #   func=mdp.bad_body_pos_z_only,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot"),
    #     "threshold": 0.25,
    #     "body_names": (),  # Set per-robot.
    #   },
    # ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "reset_init": CurriculumTermCfg(
      func=reset_initialization_curriculum,
      params={
        "event_name": "reset_base",
        "init_stages": [
          {
            "step": 0,
            "data_probability": 0.25,
            "tilt_pose_range": {
              "x": (-0.4, 0.4),
              "y": (-0.4, 0.4),
              "z": (0.00, 0.06),
              # "roll": (-0.28, 0.28),
              # "pitch": (-0.28, 0.28),
              "yaw": (-3.14, 3.14),
            },
            "tilt_velocity_range": {},
            "tilt_joint_position_range": (-0.2, 0.2),
            "tilt_joint_velocity_range": (-0.03, 0.03),
          },
          {
            "step": 5_000 * 32,
            "data_probability": 0.55,
            "tilt_pose_range": {
              "x": (-0.5, 0.5),
              "y": (-0.5, 0.5),
              "z": (0.00, 0.06),
              # "roll": (-0.38, 0.38),
              # "pitch": (-0.38, 0.38),
              "yaw": (-3.14, 3.14),
            },
            "tilt_velocity_range": {
              "roll": (-0.2, 0.2),
              "pitch": (-0.2, 0.2),
            },
            "tilt_joint_position_range": (-0.25, 0.25),
            "tilt_joint_velocity_range": (-0.05, 0.05),
          },
          {
            "step": 15_000 * 32,
            "data_probability": 0.85,
            "tilt_pose_range": {
              "x": (-0.6, 0.6),
              "y": (-0.6, 0.6),
              "z": (0.00, 0.10),
              # "roll": (-0.5, 0.5),
              # "pitch": (-0.5, 0.5),
              "yaw": (-3.14, 3.14),
            },
            "tilt_velocity_range": {
              "roll": (-0.35, 0.35),
              "pitch": (-0.35, 0.35),
              "yaw": (-0.2, 0.2),
            },
            "tilt_joint_position_range": (-0.3, 0.3),
            "tilt_joint_velocity_range": (-0.08, 0.08),
          },
        ],
      },
    ),
    "reset_push": CurriculumTermCfg(
      func=reset_push_curriculum,
      params={
        "event_name": "push_at_reset",
        # env.common_step_counter counts env steps, not iterations.
        "push_stages": [
          {
            "step": 0,
            "x": (-2.0, 2.0),
            "y": (-2.0, 2.0),
            "z": (-0.1, 0.1),
            "roll": (-0.2, 0.2),
            "pitch": (-0.2, 0.2),
            "yaw": (-0.3, 0.3),
          },
          {
            "step": 6_000 * 32,
            "x": (-4.0, 4.0),
            "y": (-4.0, 4.0),
            "z": (-0.15, 0.15),
            "roll": (-0.3, 0.3),
            "pitch": (-0.3, 0.3),
            "yaw": (-0.4, 0.4),
          },
          {
            "step": 15_000 * 32,
            "x": (-6.0, 6.0),
            "y": (-6.0, 6.0),
            "z": (-0.2, 0.2),
            "roll": (-0.4, 0.4),
            "pitch": (-0.4, 0.4),
            "yaw": (-0.3, 0.3),
          },
        ],
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(
        terrain_type="plane",
        terrain_generator=None,
      ),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=None,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      asset_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1600,
      contact_sensor_maxmatch=200,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=6,
        ls_iterations=12,
        ccd_iterations=200,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
    post_reset_freeze_steps=0,
    amp=AMPCfg(
      # Keep root z for fall-state awareness, but drop root x/y and root 6D
      # orientation so the discriminator cannot separate expert/policy too
      # easily using obvious global pose shortcuts.
      num_disc_obs_steps=2,  # 55-dim with current settings; reduces discriminator shortcutting
      asset_name="robot",
      root_body_name="LINK_BASE",
      motion_file=None,
      global_obs=False,
      root_height_obs=True,
      include_root_xy=False,
      include_root_rot=False,
    ),
  )
