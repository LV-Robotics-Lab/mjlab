"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  PM_ACTION_SCALE,
  PM_ROBOT_CFG,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def pm1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": PM_ROBOT_CFG}

  # Self-collision detection for PM1 robot
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  # Detect maximum contact force between all robot body links and ground
  # Exclude ankle_pitch and ankle_roll from contact detection
  body_contact_force_cfg = ContactSensorCfg(
    name="body_contact_force",
    primary=ContactMatch(
      mode="body",
      pattern=r"^LINK_.*$",
      entity="robot",
      exclude=(
        "LINK_ANKLE_PITCH_L",
        "LINK_ANKLE_PITCH_R",
        "LINK_ANKLE_ROLL_L",
        "LINK_ANKLE_ROLL_R",
      ),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("force", "found"),
    reduce="maxforce",
    num_slots=1,
  )
  cfg.scene.sensors = (self_collision_cfg, body_contact_force_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PM_ACTION_SCALE

  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "LINK_BASE"
  motion_cmd.body_names = (
    "LINK_BASE",
    # "LINK_HIP_PITCH_L",
    "LINK_HIP_ROLL_L",
    # "LINK_HIP_YAW_L",
    "LINK_KNEE_PITCH_L",
    # "LINK_ANKLE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    # "LINK_HIP_PITCH_R",
    "LINK_HIP_ROLL_R",
    # "LINK_HIP_YAW_R",
    "LINK_KNEE_PITCH_R",
    # "LINK_ANKLE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    "LINK_TORSO_YAW",
    # "LINK_SHOULDER_PITCH_L",
    "LINK_SHOULDER_ROLL_L",
    # "LINK_SHOULDER_YAW_L",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_YAW_L",
    # "LINK_SHOULDER_PITCH_R",
    "LINK_SHOULDER_ROLL_R",
    # "LINK_SHOULDER_YAW_R",
    "LINK_ELBOW_PITCH_R",
    "LINK_ELBOW_YAW_R",
    "LINK_HEAD_YAW",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^collision_(left|right)_foot(_toe)?$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("LINK_BASE",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R",
  )

  cfg.viewer.body_name = "LINK_BASE"

  # Fix sensor names for PM1 robot (uses different sensor names than G1)
  # PM1 uses: imu_link_linear_velocity, imu_angular_velocity
  # G1 uses: imu_lin_vel, imu_ang_vel
  if "base_lin_vel" in cfg.observations["policy"].terms:
    cfg.observations["policy"].terms["base_lin_vel"].params["sensor_name"] = "robot/imu_link_linear_velocity"
  if "base_ang_vel" in cfg.observations["policy"].terms:
    cfg.observations["policy"].terms["base_ang_vel"].params["sensor_name"] = "robot/imu_angular_velocity"
  if "base_lin_vel" in cfg.observations["critic"].terms:
    cfg.observations["critic"].terms["base_lin_vel"].params["sensor_name"] = "robot/imu_link_linear_velocity"
  if "base_ang_vel" in cfg.observations["critic"].terms:
    cfg.observations["critic"].terms["base_ang_vel"].params["sensor_name"] = "robot/imu_angular_velocity"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_policy_terms = {
      k: v
      for k, v in cfg.observations["policy"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["policy"] = ObservationGroupCfg(
      terms=new_policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
