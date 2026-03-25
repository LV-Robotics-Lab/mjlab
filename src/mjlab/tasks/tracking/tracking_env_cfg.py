"""Motion mimic task configuration.

This module defines the base configuration for motion mimic tasks.
Robot-specific configurations are located in the config/ directory.

This is a re-implementation of BeyondMimic (https://beyondmimic.github.io/).

Based on https://github.com/HybridRobotics/whole_body_tracking
Commit: f8e20c880d9c8ec7172a13d3a88a65e3a5a88448
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
  ActionTermCfg,
  CommandTermCfg,
  CurriculumTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}


def make_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base tracking task configuration."""

  ##
  # Observations
  ##

  policy_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      history_length=5,
      flatten_history_dim=True,
      clip=(-20000.0, 20000.0),  # Match ROS2 observation_clip
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
      history_length=5,
      flatten_history_dim=True,
      clip=(-20000.0, 20000.0),  # Match ROS2 observation_clip
    ),
    "actions": ObservationTermCfg(
      func=mdp.last_action,
      history_length=5,
      flatten_history_dim=True,
      clip=(-20000.0, 20000.0),  # Match ROS2 observation_clip
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      history_length=5,
      flatten_history_dim=True,
      clip=(-20000.0, 20000.0),  # Match ROS2 observation_clip
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      params={"command_name": "motion"},
      history_length=5,
      flatten_history_dim=True,
      clip=(-20000.0, 20000.0),  # Match ROS2 observation_clip
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
      history_length=5,
      flatten_history_dim=True,
      clip=(-20000.0, 20000.0),  # Match ROS2 observation_clip
    ),
    # Use scaled commands to match ROS2 observation_scale_dof_vel: 0.05
    # Original (unscaled):
    # "command": ObservationTermCfg(
    #   func=mdp.generated_commands,
    #   params={"command_name": "motion"},
    #   clip=(-20000.0, 20000.0),  # Match ROS2 observation_clip
    # ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands_with_scale,
      params={"command_name": "motion", "pos_scale": 1.0, "vel_scale": 0.05},
      clip=(-20000.0, 20000.0),  # Match ROS2 observation_clip
    ),
    # Use scaled future frames to match ROS2 observation_scale_dof_vel: 0.05
    # Original (unscaled):
    # "future_frames": ObservationTermCfg(
    #   func=mdp.future_frames_generated_commands,
    #   params={"command_name": "motion"},
    #   clip=(-20000.0, 20000.0),  # Match ROS2 observation_clip
    # ),
    "future_frames": ObservationTermCfg(
      func=mdp.future_frames_generated_commands_with_scale,
      params={"command_name": "motion", "pos_scale": 1.0, "vel_scale": 0.05},
      clip=(-20000.0, 20000.0),  # Match ROS2 observation_clip
    ),
  }

  critic_terms = {
    # Use scaled commands to match ROS2 observation_scale_dof_vel: 0.05
    # Original (unscaled):
    # "command": ObservationTermCfg(
    #   func=mdp.generated_commands,
    #   params={"command_name": "motion"},
    #   clip=(-20000.0, 20000.0),  # Match ROS2 observation_clip
    # ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands_with_scale,
      params={"command_name": "motion", "pos_scale": 1.0, "vel_scale": 0.05},
      clip=(-20000.0, 20000.0),  # Match ROS2 observation_clip
    ),
    # Use scaled future frames to match ROS2 observation_scale_dof_vel: 0.05
    # Original (unscaled):
    # "future_frames": ObservationTermCfg(
    #   func=mdp.future_frames_generated_commands,
    #   params={"command_name": "motion"},
    #   clip=(-20000.0, 20000.0),  # Match ROS2 observation_clip
    # ),
    "future_frames": ObservationTermCfg(
      func=mdp.future_frames_generated_commands_with_scale,
      params={"command_name": "motion", "pos_scale": 1.0, "vel_scale": 0.05},
      clip=(-20000.0, 20000.0),  # Match ROS2 observation_clip
    ),
    "projected_gravity_error": ObservationTermCfg(
      func=mdp.projected_gravity_error,
      params={"command_name": "motion"},
      history_length=5,
      flatten_history_dim=True,
      clip=(-20000.0, 20000.0),  # Match ROS2 observation_clip
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      history_length=5,
      flatten_history_dim=True,
      clip=(-20000.0, 20000.0),  # Match ROS2 observation_clip
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      history_length=5,
      flatten_history_dim=True,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      history_length=5,
      flatten_history_dim=True,
    ),
    "actions": ObservationTermCfg(
      func=mdp.last_action,
      history_length=5,
      flatten_history_dim=True,
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      history_length=5,
      flatten_history_dim=True,
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
      history_length=5,
      flatten_history_dim=True,
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b,
      params={"command_name": "motion"},
      history_length=5,
      flatten_history_dim=True,
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b,
      params={"command_name": "motion"},
      history_length=5,
      flatten_history_dim=True,
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      history_length=5,
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
      scale=0.5,
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "motion": MotionCommandCfg(
      asset_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      # Override in robot cfg.
      motion_file="",
      anchor_body_name="",
      body_names=(),
    )
  }

  ##
  # Events
  ##

  events: dict[str, EventTermCfg] = {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": VELOCITY_RANGE},
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set in robot cfg.
        "operation": "add",
        "field": "body_ipos",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "add_joint_default_pos": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "add",
        "field": "qpos0",
        "ranges": (-0.01, 0.01),
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

  rewards: dict[str, RewardTermCfg] = {
    "motion_global_root_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=0.2,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_global_root_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision"},
    ),
    # # 全局XY跟踪奖励：误差<=0.25给奖励1.0，超出后线性惩罚
    # motion_global_root_xy: RewTerm = term(
    #   RewTerm,
    #   func=mdp.global_xy_position_reward,
    #   weight=1.0,
    #   params={"command_name": "motion", "tolerance": 0.25, "penalty_gain": 1.0, "inside_reward": 1.0},
    # )

    # 脚部相对位置跟踪奖励：参考与机器人左右脚位置差匹配
    "feet_relative_pos": RewardTermCfg(
      func=mdp.feet_relative_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.3},
    ),

    # 重力投影跟踪奖励：跟踪参考与机器人的重力投影向量差异
    "projected_gravity_tracking": RewardTermCfg(
      func=mdp.projected_gravity_tracking_reward,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),

    # 脚踝 pitch 关节跟踪奖励
    "ankle_pitch_joint_tracking": RewardTermCfg(
      func=mdp.ankle_pitch_joint_tracking_reward,
      weight=0.25,
      params={"command_name": "motion", "std": 0.25},
    ),
    # 脚踝 roll 关节跟踪奖励
    "ankle_roll_joint_tracking": RewardTermCfg(
      func=mdp.ankle_roll_joint_tracking_reward,
      weight=0.25,
      params={"command_name": "motion", "std": 0.25},
    ),

    # 脚踝关节平滑惩罚：防止关节抖动（加速度惩罚）
    "ankle_joint_smoothness": RewardTermCfg(
      func=mdp.ankle_joint_smoothness_penalty,
      weight=5e-4,
      params={"command_name": "motion", "std": 2.0},
    ),

    # # 脚踝关节速度惩罚：限制过高的关节速度
    # ankle_joint_velocity_penalty: RewTerm = term(
    #   RewTerm,
    #   func=mdp.ankle_joint_velocity_penalty,
    #   weight=-0.1,
    #   params={"command_name": "motion", "max_vel": 5.0},
    # )

    # 脚踝关节急动度惩罚：进一步平滑运动（jerk惩罚）
    "ankle_joint_jerk_penalty": RewardTermCfg(
      func=mdp.ankle_joint_jerk_penalty,
      weight=1e-6,
      params={"command_name": "motion", "std": 5.0},
    ),

    # 脚踝关节能量消耗惩罚：惩罚高功率消耗
    # ankle_joint_power_penalty: RewTerm = term(
    #   RewTerm,
    #   func=mdp.ankle_joint_power_penalty,
    #   weight=1e-3,
    #   params={"command_name": "motion"},
    # )

    # "action_rate_l2_ankle": RewardTermCfg(
    #   func=mdp.action_rate_l2_ankle,
    #   weight=-4e-1,
    # ),

    "feet_distance_penalty": RewardTermCfg(
      func=mdp.reward_feet_distance,
      weight=1.0,
      params={
          "command_name": "motion",
      },
    ),
    
    # "foot_slip": RewardTermCfg(
    #   func=mdp.foot_slip_penalty,
    #   weight=-0.5,
    #   params={
    #     "command_name": "motion",
    #     "asset_cfg": SceneEntityCfg("robot"),
    #     "contact_threshold": 2.0,
    #     "foot_contact_sensor_names": ["force_left_foot_contact", "force_right_foot_contact"],
    #   },
    # ),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=mdp.bad_motion_body_pos_z_only,
      params={
        "command_name": "motion",
        "threshold": 0.25,  # Curriculum: starts strict, relaxes over training.
        "body_names": (),  # Set per-robot.
      },
    ),
  }

  ##
  # Curriculum
  ##

  curriculum: dict[str, CurriculumTermCfg] = {}

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainImporterCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
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
      njmax=250,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
