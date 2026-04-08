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
from mjlab.tasks.fall.mdp.curriculums import reset_push_curriculum
from mjlab.tasks.fall.mdp.events import apply_external_force_torque_axiswise_pulse
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.curriculums import (
  tracking_push_force_curriculum,
  tracking_recovery_curriculum,
  tracking_recovery_disc_weight_curriculum,
  tracking_recovery_task_weight_curriculum,
)
from mjlab.tasks.tracking.mdp.rewards import (
  recovery_success_bonus,
  recovery_time_penalty,
)
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

def make_tracking_env_cfg(
  enable_recovery_curriculum: bool = False,
) -> ManagerBasedRlEnvCfg:
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
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
      history_length=5,
      flatten_history_dim=True,
    ),
    "actions": ObservationTermCfg(
      func=mdp.last_action,
      history_length=5,
      flatten_history_dim=True,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      history_length=5,
      flatten_history_dim=True,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      params={"command_name": "motion"},
      history_length=5,
      flatten_history_dim=True,
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
      history_length=5,
      flatten_history_dim=True,
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "future_frames": ObservationTermCfg(
      func=mdp.future_frames_generated_commands,
      params={"command_name": "motion"},
    ),
  }

  critic_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "future_frames": ObservationTermCfg(
      func=mdp.future_frames_generated_commands,
      params={"command_name": "motion"},
    ),
    "projected_gravity_error": ObservationTermCfg(
      func=mdp.projected_gravity_error,
      params={"command_name": "motion"},
      history_length=5,
      flatten_history_dim=True,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      history_length=5,
      flatten_history_dim=True,
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
    "push_force_pulse": EventTermCfg(
      func=apply_external_force_torque_axiswise_pulse,
      mode="interval",
      interval_range_s=(0.0, 0.0),
      params={
        # World-frame external force range per axis.
        "force_axis_range": {
          "x": (-200.0, 200.0),
          "y": (-200.0, 200.0),
          "z": (-30.0, -30.0),
        },
        # World-frame external torque range per axis.
        "torque_axis_range": {
        },
        "duration_steps_range": (5, 20),
        "asset_cfg": SceneEntityCfg("robot"),
      },
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
  }

  ##
  # Rewards
  ##

  rewards: dict[str, RewardTermCfg] = {
    # "motion_global_root_pos": RewardTermCfg(
    #   func=mdp.motion_global_anchor_position_error_exp,
    #   weight=0.25,
    #   params={"command_name": "motion", "std": 0.3},
    # ),
    # "motion_global_root_ori": RewardTermCfg(
    #   func=mdp.motion_global_anchor_orientation_error_exp,
    #   weight=0.25,
    #   params={"command_name": "motion", "std": 0.4},
    # ),
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
    # Recovery-only: reduce vulnerable-body contact force.
    "recovery_reduce_contact_force": RewardTermCfg(
      func=mdp.recovery_reduce_contact_force_weighted,
      weight=0.01,
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
          "LINK_SHOULDER_ROLL_L",
          "LINK_SHOULDER_ROLL_R",
          "LINK_SHOULDER_YAW_L",
          "LINK_SHOULDER_YAW_R",
        ),
        "high_weight": 10.0,
        "medium_weight": 2.0,
        "low_weight": 0.5,
        "alpha": 0.3,
      },
    ),
    # Recovery-only: penalize body height drop vs motion reference (link configurable).
    "recovery_body_height": RewardTermCfg(
      func=mdp.recovery_body_height_penalty,
      weight=1.0,
      params={
        "body_name": "LINK_TORSO_YAW",
        "command_name": "motion",
        "asset_cfg": SceneEntityCfg("robot"),
        "penalty_scale": 1.0,
      },
    ),
    # One-shot penalty the same step recovery is entered (discourage diving into recovery).
    # "recovery_entry_penalty": RewardTermCfg(
    #   func=mdp.recovery_entry_penalty_reward,
    #   weight=10.0,
    #   params={},
    # ),
    # # Recovery-only: per-step time pressure, encourages earlier completion.
    # "recovery_time_penalty": RewardTermCfg(
    #   func=recovery_time_penalty,
    #   weight=10.0,
    #   params={"per_step_penalty": 0.02},
    # ),
    # Recovery-only: one-step bonus when recovery exits before timeout.
    # "recovery_success_bonus": RewardTermCfg(
    #   func=recovery_success_bonus,
    #   weight=5.0,
    #   params={"bonus_scale": 100.0},
    # ),
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
    # "projected_gravity_tracking": RewardTermCfg(
    #   func=mdp.projected_gravity_tracking_reward,
    #   weight=1.0,
    #   params={"command_name": "motion", "std": 1.0},
    # ),

    # 脚踝 pitch 关节跟踪奖励
    # "ankle_pitch_joint_tracking": RewardTermCfg(
    #   func=mdp.ankle_pitch_joint_tracking_reward,
    #   weight=0.25,
    #   params={"command_name": "motion", "std": 0.25},
    # ),
    # 脚踝 roll 关节跟踪奖励
    # "ankle_roll_joint_tracking": RewardTermCfg(
    #   func=mdp.ankle_roll_joint_tracking_reward,
    #   weight=0.25,
    #   params={"command_name": "motion", "std": 0.25},
    # ),

    # 脚踝关节平滑惩罚：防止关节抖动（加速度惩罚）
    "ankle_joint_smoothness": RewardTermCfg(
      func=mdp.ankle_joint_smoothness_penalty,
      weight=5e-4,
      params={"command_name": "motion", "std": 2.0},
    ),

    # # # 脚踝关节速度惩罚：限制过高的关节速度
    # # ankle_joint_velocity_penalty: RewTerm = term(
    # #   RewTerm,
    # #   func=mdp.ankle_joint_velocity_penalty,
    # #   weight=-0.1,
    # #   params={"command_name": "motion", "max_vel": 5.0},
    # # )

    # 脚踝关节急动度惩罚：进一步平滑运动（jerk惩罚）
    "ankle_joint_jerk_penalty": RewardTermCfg(
      func=mdp.ankle_joint_jerk_penalty,
      weight=1e-6,
      params={"command_name": "motion", "std": 5.0},
    ),

    # # 脚踝关节能量消耗惩罚：惩罚高功率消耗
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

    # "feet_distance_penalty": RewardTermCfg(
    #   func=mdp.reward_feet_distance,
    #   weight=1.0,
    #   params={
    #       "command_name": "motion",
    #   },
    # ),
    
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
    # Pre-recovery（recovery 课程未开启）：anchor / ee 等仍按原阈值终局。
    # Recovery 开启后：下列条目改为触发 recovery（不终局）。
    "anchor_pos": TerminationTermCfg(
      func=mdp.recovery_or_terminate_bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.recovery_or_terminate_bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=mdp.recovery_or_terminate_bad_motion_body_pos_z_only,
      params={
        "command_name": "motion",
        "threshold": 0.25,
        "body_names": (),  # Set per-robot.
      },
    ),
    # Recovery 内：仍用原先 anchor z / anchor ori / ee body z 判定是否跟上；超时终局。
    "recovery_mismatch": TerminationTermCfg(
      func=mdp.recovery_mismatch_after_duration,
      params={
        "command_name": "motion",
        "recovery_duration_s": 6.0,
        "anchor_pos_threshold": 0.25,
        "ee_body_pos_threshold": 0.25,
        "body_names": (),  # Set per-robot.
        "asset_cfg": SceneEntityCfg("robot"),
        "success_stable_steps": 4,
        "success_hysteresis_decay": 1,
      },
    ),
  }

  curriculum = {
    "tracking_push_force": CurriculumTermCfg(
      func=tracking_push_force_curriculum,
      params={
        "event_name": "push_force_pulse",
        "force_stages": [
          {
            "step": 0,
            "x": (0, 0),
            "y": (0, 0),
            "z": (0, 0),
            "duration_steps_range": (0, 0),
          },
          {
            "step": 8_000 * 32,
            "x": (-200.0, 200.0),
            "y": (-200.0, 200.0),
            "z": (-25.0, -25.0),
            "duration_steps_range": (0, 8),
          },
        ],
      },
    ),
    # "tracking_push_robot": CurriculumTermCfg(
    #   func=reset_push_curriculum,
    #   params={
    #     "event_name": "push_robot",
    #     "push_stages": [
    #       {
    #         "step": 0,
    #         "x": (-0.5, 0.5),
    #         "y": (-0.5, 0.5),
    #         "z": (-0.2, 0.2),
    #         "roll": (-0.3, 0.3),
    #         "pitch": (-0.3, 0.3),
    #         "yaw": (-0.4, 0.4),
    #       },
    #       {
    #         "step": 8_000 * 32,
    #         "x": (-1.2, 1.2),
    #         "y": (-1.2, 1.2),
    #         "z": (-0.35, 0.35),
    #         "roll": (-0.42, 0.42),
    #         "pitch": (-0.42, 0.42),
    #         "yaw": (-0.55, 0.55),
    #       },
    #       {
    #         "step": 11_000 * 32,
    #         "x": (-2.4, 2.4),
    #         "y": (-2.4, 2.4),
    #         "z": (-0.65, 0.65),
    #         "roll": (-0.7, 0.7),
    #         "pitch": (-0.7, 0.7),
    #         "yaw": (-0.9, 0.9),
    #       },
    #       {
    #         "step": 15_000 * 32,
    #         "x": (-4, 4),
    #         "y": (-4, 4),
    #         "z": (-1, 1),
    #         "roll": (-1, 1),
    #         "pitch": (-1, 1),
    #         "yaw": (-1.2, 1.2),
    #       },
    #     ],
    #   },
    # ),
  }
  if enable_recovery_curriculum:
    curriculum["tracking_recovery"] = CurriculumTermCfg(
      func=tracking_recovery_curriculum,
      params={
        "recovery_start_common_step": 8_000 * 32,
      },
    )
    curriculum["tracking_recovery_disc_weight"] = CurriculumTermCfg(
      func=tracking_recovery_disc_weight_curriculum,
      params={
        "stages": [
          {"step": 0, "scale": 1.0},
          {"step": 12_000 * 32, "scale": 1.5},
          {"step": 18_000 * 32, "scale": 2.0},
        ],
      },
    )
    curriculum["tracking_recovery_task_weight"] = CurriculumTermCfg(
      func=tracking_recovery_task_weight_curriculum,
      params={
        "stages": [
          {"step": 0, "scale": 2.0},
          {"step": 14_000 * 32, "scale": 1.2},
          {"step": 18_000 * 32, "scale": 0.6},
          {"step": 22_000 * 32, "scale": 0.4},
        ],
      },
    )

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
