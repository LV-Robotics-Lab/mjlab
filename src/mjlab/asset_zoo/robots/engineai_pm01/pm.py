import isaaclab.sim as sim_utils
from engineai_boxing_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# from engineai_boxing_lab.assets.torque_speed_motor import TorqueSpeedMotorActuatorCfg
# TODO: Try ImplicitActuator firstly

# =============================================================================
# PM01 Robot Physical Parameters
# =============================================================================

# Armature values for different joint types
ARMATURE_Q90 = 0.0453  # High-torque joints (HIP_PITCH, KNEE_PITCH)
ARMATURE_Q25 = 0.0067  # Low-torque joints (HIP_YAW, ANKLE, SHOULDER, ELBOW, HEAD)

EFFORT_LIMIT_Q90 = 164.0  # High-torque joints (HIP_PITCH, KNEE_PITCH)
EFFORT_LIMIT_Q25 = 52.0  # Low-torque joints (HIP_YAW, ANKLE, SHOULDER, ELBOW, HEAD)

VELOCITY_LIMIT_Q90 = 26.3  # High-torque joints (HIP_PITCH, KNEE_PITCH)
VELOCITY_LIMIT_Q25 = 35.2  # Low-torque joints (HIP_YAW, ANKLE, SHOULDER, ELBOW, HEAD)

# Control parameters
NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535  # 10Hz natural frequency
DAMPING_RATIO = 2.0  # Critical damping ratio

# Calculate stiffness and damping based on natural frequency and damping ratio
STIFFNESS_Q90 = ARMATURE_Q90 * NATURAL_FREQ**2
STIFFNESS_Q25 = ARMATURE_Q25 * NATURAL_FREQ**2
DAMPING_Q90 = 2.0 * DAMPING_RATIO * ARMATURE_Q90 * NATURAL_FREQ
DAMPING_Q25 = 2.0 * DAMPING_RATIO * ARMATURE_Q25 * NATURAL_FREQ

# torque limit = -3.14/30 * 1 * speed + 260
TNCURVE_SLOPE_Q90 = -3.14 / 30 * 1
TNCURVE_SLOPE_Q25 = -3.14 / 30 * 0.18
TNCURVE_INTERCEPT_Q90 = 260
TNCURVE_INTERCEPT_Q25 = 67

# =============================================================================
# PM01 Robot Configuration
# =============================================================================

ENGINEAI_PM01_CFG = ArticulationCfg(
  spawn=sim_utils.UrdfFileCfg(
    fix_base=False,
    replace_cylinders_with_capsules=False,
    asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/engineai/pm01_description/urdf/serial_pm_v2_boxing.urdf",
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
      disable_gravity=False,
      retain_accelerations=False,
      linear_damping=0.0,
      angular_damping=0.0,
      max_linear_velocity=1000.0,
      max_angular_velocity=1000.0,
      max_depenetration_velocity=1.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
      enabled_self_collisions=True,
      solver_position_iteration_count=8,
      solver_velocity_iteration_count=4,
    ),
    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
      gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    ),
  ),
  init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.76),
    joint_pos={
      "J00_HIP_PITCH_L": -0.6855795466,
      "J01_HIP_ROLL_L": 0.1943894958,
      "J02_HIP_YAW_L": -0.1694991227,
      "J03_KNEE_PITCH_L": 0.8763085749,
      "J04_ANKLE_PITCH_L": -0.2966122755,
      "J05_ANKLE_ROLL_L": -0.1642948528,
      "J06_HIP_PITCH_R": -0.5869977636,
      "J07_HIP_ROLL_R": -0.1031214416,
      "J08_HIP_YAW_R": 0.0722257223,
      "J09_KNEE_PITCH_R": 0.8448515456,
      "J10_ANKLE_PITCH_R": -0.3247703797,
      "J11_ANKLE_ROLL_R": 0.1247136230,
      "J12_WAIST_YAW": 0.3588285899,
      "J13_SHOULDER_PITCH_L": -0.4540878383,
      "J14_SHOULDER_ROLL_L": 0.3463471061,
      "J15_SHOULDER_YAW_L": -0.3236981137,
      "J16_ELBOW_PITCH_L": -1.8448246279,
      "J17_ELBOW_YAW_L": 0.3379777493,
      "J18_SHOULDER_PITCH_R": -0.9340156194,
      "J19_SHOULDER_ROLL_R": -0.2239855084,
      "J20_SHOULDER_YAW_R": 0.5954717404,
      "J21_ELBOW_PITCH_R": -2.0112772205,
      "J22_ELBOW_YAW_R": -0.3283864538,
      "J23_HEAD_YAW": 0.4827054124,
    },
    joint_vel={".*": 0.0},
  ),
  soft_joint_pos_limit_factor=0.9,
  actuators={
    "legs": ImplicitActuatorCfg(
      joint_names_expr=[
        ".*_HIP_PITCH.*",
        ".*_HIP_ROLL.*",
        ".*_HIP_YAW.*",
        ".*_KNEE_PITCH.*",
      ],
      # tn_slope=TNCURVE_SLOPE_Q90,
      # tn_intercept=TNCURVE_INTERCEPT_Q90,
      effort_limit_sim={
        ".*_HIP_PITCH.*": EFFORT_LIMIT_Q90,
        ".*_HIP_ROLL.*": EFFORT_LIMIT_Q90,
        ".*_HIP_YAW.*": EFFORT_LIMIT_Q25,
        ".*_KNEE_PITCH.*": EFFORT_LIMIT_Q90,
      },
      velocity_limit_sim={
        ".*_HIP_PITCH.*": VELOCITY_LIMIT_Q90,
        ".*_HIP_ROLL.*": VELOCITY_LIMIT_Q90,
        ".*_HIP_YAW.*": VELOCITY_LIMIT_Q25,
        ".*_KNEE_PITCH.*": VELOCITY_LIMIT_Q90,
      },
      stiffness={
        ".*_HIP_PITCH.*": STIFFNESS_Q90,
        ".*_HIP_ROLL.*": STIFFNESS_Q90,
        ".*_HIP_YAW.*": STIFFNESS_Q25,
        ".*_KNEE_PITCH.*": STIFFNESS_Q90,
      },
      damping={
        ".*_HIP_PITCH.*": DAMPING_Q90,
        ".*_HIP_ROLL.*": DAMPING_Q90,
        ".*_HIP_YAW.*": DAMPING_Q25,
        ".*_KNEE_PITCH.*": DAMPING_Q90,
      },
      armature={
        ".*_HIP_PITCH.*": ARMATURE_Q90,
        ".*_HIP_ROLL.*": ARMATURE_Q90,
        ".*_HIP_YAW.*": ARMATURE_Q25,
        ".*_KNEE_PITCH.*": ARMATURE_Q90,
      },
      friction={
        ".*_HIP_PITCH.*": 0.1,  # Static friction for hip pitch
        ".*_HIP_ROLL.*": 0.1,  # Static friction for hip roll
        ".*_HIP_YAW.*": 0.05,  # Lower friction for yaw joints
        ".*_KNEE_PITCH.*": 0.1,  # Static friction for knee
      },
      dynamic_friction={
        ".*_HIP_PITCH.*": 0.08,
        ".*_HIP_ROLL.*": 0.08,
        ".*_HIP_YAW.*": 0.04,
        ".*_KNEE_PITCH.*": 0.08,
      },
      viscous_friction={
        ".*_HIP_PITCH.*": 0.01,
        ".*_HIP_ROLL.*": 0.01,
        ".*_HIP_YAW.*": 0.005,
        ".*_KNEE_PITCH.*": 0.01,
      },
    ),
    "feet": ImplicitActuatorCfg(
      joint_names_expr=[
        ".*_ANKLE_PITCH.*",
        ".*_ANKLE_ROLL.*",
      ],
      # tn_slope=TNCURVE_SLOPE_Q25,
      # tn_intercept=TNCURVE_INTERCEPT_Q25,
      effort_limit_sim={
        ".*_ANKLE_PITCH.*": EFFORT_LIMIT_Q25,
        ".*_ANKLE_ROLL.*": EFFORT_LIMIT_Q25,
      },
      velocity_limit_sim={
        ".*_ANKLE_PITCH.*": VELOCITY_LIMIT_Q25,
        ".*_ANKLE_ROLL.*": VELOCITY_LIMIT_Q25,
      },
      stiffness={
        ".*_ANKLE_PITCH.*": STIFFNESS_Q25,
        ".*_ANKLE_ROLL.*": STIFFNESS_Q25,
      },
      damping={
        ".*_ANKLE_PITCH.*": 0.5,
        ".*_ANKLE_ROLL.*": 0.5,
      },
      armature={
        ".*_ANKLE_ROLL.*": ARMATURE_Q25,
        ".*_ANKLE_PITCH.*": ARMATURE_Q25,
      },
      friction={
        ".*_ANKLE_PITCH.*": 0.15,  # Higher friction for ankle stability
        ".*_ANKLE_ROLL.*": 0.12,  # Moderate friction for ankle roll
      },
      dynamic_friction={
        ".*_ANKLE_PITCH.*": 0.12,
        ".*_ANKLE_ROLL.*": 0.10,
      },
      viscous_friction={
        ".*_ANKLE_PITCH.*": 0.015,
        ".*_ANKLE_ROLL.*": 0.012,
      },
    ),
    "waist_yaw": ImplicitActuatorCfg(
      joint_names_expr=["J12_WAIST_YAW"],
      # tn_slope=TNCURVE_SLOPE_Q25,
      # tn_intercept=TNCURVE_INTERCEPT_Q25,
      effort_limit_sim=EFFORT_LIMIT_Q25,
      velocity_limit_sim=VELOCITY_LIMIT_Q25,
      stiffness=STIFFNESS_Q25,
      damping=DAMPING_Q25,
      armature=ARMATURE_Q25,
      friction=0.08,  # Moderate friction for waist rotation
      dynamic_friction=0.06,
      viscous_friction=0.008,
    ),
    "arms": ImplicitActuatorCfg(
      joint_names_expr=[
        ".*_SHOULDER_PITCH.*",
        ".*_SHOULDER_ROLL.*",
        ".*_SHOULDER_YAW.*",
        ".*_ELBOW_PITCH.*",
        ".*_ELBOW_YAW.*",
      ],
      # tn_slope=TNCURVE_SLOPE_Q25,
      # tn_intercept=TNCURVE_INTERCEPT_Q25,
      effort_limit_sim={
        ".*_SHOULDER_PITCH.*": EFFORT_LIMIT_Q25,
        ".*_SHOULDER_ROLL.*": EFFORT_LIMIT_Q25,
        ".*_SHOULDER_YAW.*": EFFORT_LIMIT_Q25,
        ".*_ELBOW_PITCH.*": EFFORT_LIMIT_Q25,
        ".*_ELBOW_YAW.*": EFFORT_LIMIT_Q25,
      },
      velocity_limit_sim={
        ".*_SHOULDER_PITCH.*": VELOCITY_LIMIT_Q25,
        ".*_SHOULDER_ROLL.*": VELOCITY_LIMIT_Q25,
        ".*_SHOULDER_YAW.*": VELOCITY_LIMIT_Q25,
        ".*_ELBOW_PITCH.*": VELOCITY_LIMIT_Q25,
        ".*_ELBOW_YAW.*": VELOCITY_LIMIT_Q25,
      },
      stiffness={
        ".*_SHOULDER_PITCH.*": STIFFNESS_Q25,
        ".*_SHOULDER_ROLL.*": STIFFNESS_Q25,
        ".*_SHOULDER_YAW.*": STIFFNESS_Q25,
        ".*_ELBOW_PITCH.*": STIFFNESS_Q25,
        ".*_ELBOW_YAW.*": STIFFNESS_Q25,
      },
      damping={
        ".*_SHOULDER_PITCH.*": DAMPING_Q25,
        ".*_SHOULDER_ROLL.*": DAMPING_Q25,
        ".*_SHOULDER_YAW.*": DAMPING_Q25,
        ".*_ELBOW_PITCH.*": DAMPING_Q25,
        ".*_ELBOW_YAW.*": DAMPING_Q25,
      },
      armature={
        ".*_SHOULDER_PITCH.*": ARMATURE_Q25,
        ".*_SHOULDER_ROLL.*": ARMATURE_Q25,
        ".*_SHOULDER_YAW.*": ARMATURE_Q25,
        ".*_ELBOW_PITCH.*": ARMATURE_Q25,
        ".*_ELBOW_YAW.*": ARMATURE_Q25,
      },
      friction={
        ".*_SHOULDER_PITCH.*": 0.12,  # Higher friction for shoulder stability
        ".*_SHOULDER_ROLL.*": 0.10,
        ".*_SHOULDER_YAW.*": 0.08,  # Lower friction for yaw joints
        ".*_ELBOW_PITCH.*": 0.08,  # Moderate friction for elbows
        ".*_ELBOW_YAW.*": 0.06,
      },
      dynamic_friction={
        ".*_SHOULDER_PITCH.*": 0.10,
        ".*_SHOULDER_ROLL.*": 0.08,
        ".*_SHOULDER_YAW.*": 0.06,
        ".*_ELBOW_PITCH.*": 0.06,
        ".*_ELBOW_YAW.*": 0.05,
      },
      viscous_friction={
        ".*_SHOULDER_PITCH.*": 0.012,
        ".*_SHOULDER_ROLL.*": 0.010,
        ".*_SHOULDER_YAW.*": 0.008,
        ".*_ELBOW_PITCH.*": 0.008,
        ".*_ELBOW_YAW.*": 0.006,
      },
    ),
    "head": ImplicitActuatorCfg(
      joint_names_expr=["J23_HEAD_YAW"],
      # tn_slope=TNCURVE_SLOPE_Q25,
      # tn_intercept=TNCURVE_INTERCEPT_Q25,
      effort_limit_sim=EFFORT_LIMIT_Q25,
      velocity_limit_sim=VELOCITY_LIMIT_Q25,
      stiffness=STIFFNESS_Q25,
      damping=DAMPING_Q25,
      armature=ARMATURE_Q25,
      friction=0.05,  # Light friction for head movement
      dynamic_friction=0.04,
      viscous_friction=0.005,
    ),
  },
)


def calculate_action_scales(robot_cfg):
  """Calculate action scales for each joint based on effort limits and stiffness."""
  action_scales = {}

  for _actuator_name, actuator in robot_cfg.actuators.items():
    effort_limits = actuator.effort_limit_sim
    stiffness_values = actuator.stiffness
    joint_names = actuator.joint_names_expr

    # Handle both scalar and dictionary values
    if not isinstance(effort_limits, dict):
      effort_limits = {name: effort_limits for name in joint_names}
    if not isinstance(stiffness_values, dict):
      stiffness_values = {name: stiffness_values for name in joint_names}

    # Calculate action scale for each joint
    for joint_name in joint_names:
      if joint_name in effort_limits and joint_name in stiffness_values:
        if stiffness_values[joint_name] > 0:  # Avoid division by zero
          action_scales[joint_name] = (
            0.25 * effort_limits[joint_name] / stiffness_values[joint_name]
          )
        else:
          # Fallback for joints with zero stiffness
          action_scales[joint_name] = 0.25

  return action_scales


# Calculate action scales for PM01 robot
PM01_ACTION_SCALE = calculate_action_scales(ENGINEAI_PM01_CFG)
