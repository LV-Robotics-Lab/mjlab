# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from engineai_boxing_lab.assets import ISAACLAB_ASSETS_DATA_DIR

# from engineai_boxing_lab.assets.torque_speed_motor import TorqueSpeedMotorActuatorCfg
# TODO: Try ImplicitActuator firstly

# =============================================================================
# T800 Robot Physical Parameters
# =============================================================================

# Armature values for different joint types
ARMATURE_Q300H_L = 0.2427264  # High-torque joints (HIP_PITCH, KNEE_PITCH)
ARMATURE_Q300H = 0.14110848  # Low-torque joints (HIP_YAW, ANKLE, SHOULDER, ELBOW, HEAD)
ARMATURE_Q200H = 0.0448737
ARMATURE_Q50H = 0.0354625
ARMATURE_Q25H = 0.00671625

EFFORT_LIMIT_Q300H_L = 415  # High-torque joints (HIP_PITCH, KNEE_PITCH)
EFFORT_LIMIT_Q300H = 370  # Low-torque joints (HIP_YAW, ANKLE, SHOULDER, ELBOW, HEAD)
EFFORT_LIMIT_Q200H = 222  # Low-torque joints (HIP_YAW, ANKLE, SHOULDER, ELBOW, HEAD)
EFFORT_LIMIT_Q50H = 160  # Low-torque joints (HIP_YAW, ANKLE, SHOULDER, ELBOW, HEAD)
EFFORT_LIMIT_Q25H = 52
VELOCITY_LIMIT_Q300H_L = 25.96  # High-torque joints (HIP_PITCH, KNEE_PITCH)
VELOCITY_LIMIT_Q300H = (
    25.31  # Low-torque joints (HIP_YAW, ANKLE, SHOULDER, ELBOW, HEAD)
)
VELOCITY_LIMIT_Q200H = (
    23.19  # Low-torque joints (HIP_YAW, ANKLE, SHOULDER, ELBOW, HEAD)
)
VELOCITY_LIMIT_Q50H = 33.51  # Low-torque joints (HIP_YAW, ANKLE, SHOULDER, ELBOW, HEAD)
VELOCITY_LIMIT_Q25H = 35.2
# Control parameters
NATURAL_FREQ = 6.0 * 2.0 * 3.1415926535  # 6Hz natural frequency
DAMPING_RATIO = 1.2  # 1.2 damping ratio

# Calculate stiffness and damping based on natural frequency and damping ratio
STIFFNESS_Q300H_L = ARMATURE_Q300H_L * NATURAL_FREQ**2
STIFFNESS_Q300H = ARMATURE_Q300H * NATURAL_FREQ**2
STIFFNESS_Q200H = ARMATURE_Q200H * NATURAL_FREQ**2
STIFFNESS_Q50H = ARMATURE_Q50H * NATURAL_FREQ**2
STIFFNESS_Q25H = ARMATURE_Q25H * NATURAL_FREQ**2
DAMPING_Q300H_L = 2.0 * DAMPING_RATIO * ARMATURE_Q300H_L * NATURAL_FREQ
DAMPING_Q300H = 2.0 * DAMPING_RATIO * ARMATURE_Q300H * NATURAL_FREQ
DAMPING_Q200H = 2.0 * DAMPING_RATIO * ARMATURE_Q200H * NATURAL_FREQ
DAMPING_Q50H = 2.0 * DAMPING_RATIO * ARMATURE_Q50H * NATURAL_FREQ
DAMPING_Q25H = 2.0 * DAMPING_RATIO * ARMATURE_Q25H * NATURAL_FREQ

# torque limit = -3.14/30 * 1 * speed + 260
TNCURVE_SLOPE_Q90 = -3.14 / 30 * 1
TNCURVE_SLOPE_Q25 = -3.14 / 30 * 0.18
TNCURVE_INTERCEPT_Q90 = 260
TNCURVE_INTERCEPT_Q25 = 67

# =============================================================================
# T800 Robot Configuration
# =============================================================================

ENGINEAI_T800_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/engineai/T800/urdf/t800_20251113.urdf",
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
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            "J00_HIP_PITCH_L": -0.7732778293,
            "J01_HIP_ROLL_L": 0.2109002427,
            "J02_HIP_YAW_L": -0.2140328236,
            "J03_KNEE_PITCH_L": 0.9002003037,
            "J04_ANKLE_PITCH_L": -0.2894360952,
            "J05_ANKLE_ROLL_L": -0.1873647404,
            "J06_HIP_PITCH_R": -0.6147637016,
            "J07_HIP_ROLL_R": -0.1066712524,
            "J08_HIP_YAW_R": 0.0556522214,
            "J09_KNEE_PITCH_R": 0.8299468884,
            "J10_ANKLE_PITCH_R": -0.3985402688,
            "J11_ANKLE_ROLL_R": 0.1808947799,
            "J12_TORSO_YAW": 0.0235501909,
            "J13_SHOULDER_PITCH_L": -0.6900132027,
            "J14_SHOULDER_ROLL_L": 0.9303021872,
            "J15_SHOULDER_YAW_L": -0.1659253576,
            "J16_ELBOW_PITCH_L": -1.9041238125,
            "J17_ELBOW_YAW_L": -0.2584387427,
            "J20_SHOULDER_PITCH_R": -1.1696804212,
            "J21_SHOULDER_ROLL_R": -0.1520707048,
            "J22_SHOULDER_YAW_R": 0.6273249496,
            "J23_ELBOW_PITCH_R": -1.8817574626,
            "J24_ELBOW_YAW_R": 0.2462114231,
            "J27_HEAD_PITCH": 0.1889251056,
            "J28_HEAD_YAW": 0.6451696690,
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
                ".*_HIP_PITCH.*": EFFORT_LIMIT_Q300H_L,
                ".*_HIP_ROLL.*": EFFORT_LIMIT_Q300H,
                ".*_HIP_YAW.*": EFFORT_LIMIT_Q200H,
                ".*_KNEE_PITCH.*": EFFORT_LIMIT_Q300H_L,
            },
            velocity_limit_sim={
                ".*_HIP_PITCH.*": VELOCITY_LIMIT_Q300H_L,
                ".*_HIP_ROLL.*": VELOCITY_LIMIT_Q300H,
                ".*_HIP_YAW.*": VELOCITY_LIMIT_Q200H,
                ".*_KNEE_PITCH.*": VELOCITY_LIMIT_Q300H_L,
            },
            stiffness={
                ".*_HIP_PITCH.*": STIFFNESS_Q300H_L,
                ".*_HIP_ROLL.*": STIFFNESS_Q300H,
                ".*_HIP_YAW.*": STIFFNESS_Q200H,
                ".*_KNEE_PITCH.*": STIFFNESS_Q300H_L,
            },
            damping={
                ".*_HIP_PITCH.*": DAMPING_Q300H_L,
                ".*_HIP_ROLL.*": DAMPING_Q300H,
                ".*_HIP_YAW.*": DAMPING_Q200H,
                ".*_KNEE_PITCH.*": DAMPING_Q300H_L,
            },
            armature={
                ".*_HIP_PITCH.*": ARMATURE_Q300H_L,
                ".*_HIP_ROLL.*": ARMATURE_Q300H,
                ".*_HIP_YAW.*": ARMATURE_Q200H,
                ".*_KNEE_PITCH.*": ARMATURE_Q300H_L,
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
                ".*_ANKLE_PITCH.*": EFFORT_LIMIT_Q50H,
                ".*_ANKLE_ROLL.*": EFFORT_LIMIT_Q50H,
            },
            velocity_limit_sim={
                ".*_ANKLE_PITCH.*": VELOCITY_LIMIT_Q50H,
                ".*_ANKLE_ROLL.*": VELOCITY_LIMIT_Q50H,
            },
            stiffness={
                ".*_ANKLE_PITCH.*": STIFFNESS_Q50H,
                ".*_ANKLE_ROLL.*": STIFFNESS_Q50H,
            },
            damping={
                ".*_ANKLE_PITCH.*": DAMPING_Q50H,
                ".*_ANKLE_ROLL.*": DAMPING_Q50H,
            },
            armature={
                ".*_ANKLE_ROLL.*": ARMATURE_Q50H,
                ".*_ANKLE_PITCH.*": ARMATURE_Q50H,
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
        "torso_yaw": ImplicitActuatorCfg(
            joint_names_expr=["J12_TORSO_YAW"],
            # tn_slope=TNCURVE_SLOPE_Q25,
            # tn_intercept=TNCURVE_INTERCEPT_Q25,
            effort_limit_sim=EFFORT_LIMIT_Q200H,
            velocity_limit_sim=VELOCITY_LIMIT_Q200H,
            stiffness=STIFFNESS_Q200H,
            damping=DAMPING_Q200H,
            armature=ARMATURE_Q200H,
            friction=0.08,  # Moderate friction for torso rotation
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
                ".*_SHOULDER_PITCH.*": EFFORT_LIMIT_Q50H,
                ".*_SHOULDER_ROLL.*": EFFORT_LIMIT_Q50H,
                ".*_SHOULDER_YAW.*": EFFORT_LIMIT_Q50H,
                ".*_ELBOW_PITCH.*": EFFORT_LIMIT_Q50H,
                ".*_ELBOW_YAW.*": EFFORT_LIMIT_Q25H,
            },
            velocity_limit_sim={
                ".*_SHOULDER_PITCH.*": VELOCITY_LIMIT_Q50H,
                ".*_SHOULDER_ROLL.*": VELOCITY_LIMIT_Q50H,
                ".*_SHOULDER_YAW.*": VELOCITY_LIMIT_Q50H,
                ".*_ELBOW_PITCH.*": VELOCITY_LIMIT_Q50H,
                ".*_ELBOW_YAW.*": VELOCITY_LIMIT_Q25H,
            },
            stiffness={
                ".*_SHOULDER_PITCH.*": STIFFNESS_Q50H,
                ".*_SHOULDER_ROLL.*": STIFFNESS_Q50H,
                ".*_SHOULDER_YAW.*": STIFFNESS_Q50H,
                ".*_ELBOW_PITCH.*": STIFFNESS_Q50H,
                ".*_ELBOW_YAW.*": STIFFNESS_Q25H,
            },
            damping={
                ".*_SHOULDER_PITCH.*": DAMPING_Q50H,
                ".*_SHOULDER_ROLL.*": DAMPING_Q50H,
                ".*_SHOULDER_YAW.*": DAMPING_Q50H,
                ".*_ELBOW_PITCH.*": DAMPING_Q50H,
                ".*_ELBOW_YAW.*": DAMPING_Q25H,
            },
            armature={
                ".*_SHOULDER_PITCH.*": ARMATURE_Q50H,
                ".*_SHOULDER_ROLL.*": ARMATURE_Q50H,
                ".*_SHOULDER_YAW.*": ARMATURE_Q50H,
                ".*_ELBOW_PITCH.*": ARMATURE_Q50H,
                ".*_ELBOW_YAW.*": ARMATURE_Q25H,
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
            joint_names_expr=[
                "J27_HEAD_PITCH",
                "J28_HEAD_YAW",
            ],
            # tn_slope=TNCURVE_SLOPE_Q25,
            # tn_intercept=TNCURVE_INTERCEPT_Q25,
            effort_limit_sim={
                "J27_HEAD_PITCH": EFFORT_LIMIT_Q25H,
                "J28_HEAD_YAW": EFFORT_LIMIT_Q25H,
            },
            velocity_limit_sim={
                "J27_HEAD_PITCH": VELOCITY_LIMIT_Q25H,
                "J28_HEAD_YAW": VELOCITY_LIMIT_Q25H,
            },
            stiffness={
                "J27_HEAD_PITCH": STIFFNESS_Q25H,
                "J28_HEAD_YAW": STIFFNESS_Q25H,
            },
            damping={
                "J27_HEAD_PITCH": DAMPING_Q25H,
                "J28_HEAD_YAW": DAMPING_Q25H,
            },
            armature={
                "J27_HEAD_PITCH": ARMATURE_Q25H,
                "J28_HEAD_YAW": ARMATURE_Q25H,
            },
            friction={
                "J27_HEAD_PITCH": 0.05,  # Light friction for head movement
                "J28_HEAD_YAW": 0.05,
            },
            dynamic_friction={
                "J27_HEAD_PITCH": 0.04,
                "J28_HEAD_YAW": 0.04,
            },
            viscous_friction={
                "J27_HEAD_PITCH": 0.005,
                "J28_HEAD_YAW": 0.005,
            },
        ),
    },
)


def calculate_action_scales(robot_cfg):
    """Calculate action scales for each joint based on effort limits and stiffness."""
    action_scales = {}

    for actuator_name, actuator in robot_cfg.actuators.items():
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
            action_scales[joint_name] = 0.25
        #     if joint_name in effort_limits and joint_name in stiffness_values:
        #         if stiffness_values[joint_name] > 0:  # Avoid division by zero
        #             action_scales[joint_name] = (
        #                 0.1 * effort_limits[joint_name] / stiffness_values[joint_name]
        #             )
        #         else:
        #             # Fallback for joints with zero stiffness
        #             action_scales[joint_name] = 0.25

    return action_scales


# Calculate action scales for T800 robot
T800_ACTION_SCALE = calculate_action_scales(ENGINEAI_T800_CFG)

T800_ACTION_OFFSET = {
    "J00_HIP_PITCH_L": -0.0,
    "J01_HIP_ROLL_L": 0.0,
    "J02_HIP_YAW_L": 0.0,
    "J03_KNEE_PITCH_L": 0.0,
    "J04_ANKLE_PITCH_L": -0.0,
    "J05_ANKLE_ROLL_L": 0.0,
    "J06_HIP_PITCH_R": -0.0,
    "J07_HIP_ROLL_R": 0.0,
    "J08_HIP_YAW_R": 0.0,
    "J09_KNEE_PITCH_R": 0.0,
    "J10_ANKLE_PITCH_R": -0.0,
    "J11_ANKLE_ROLL_R": 0.0,
    "J12_TORSO_YAW": 0.0,
    "J13_SHOULDER_PITCH_L": -0.0,
    "J14_SHOULDER_ROLL_L": 0.0,
    "J15_SHOULDER_YAW_L": 0.0,
    "J16_ELBOW_PITCH_L": -0.0,
    "J17_ELBOW_YAW_L": 0.0,
    "J20_SHOULDER_PITCH_R": -0.0,
    "J21_SHOULDER_ROLL_R": 0.0,
    "J22_SHOULDER_YAW_R": 0.0,
    "J23_ELBOW_PITCH_R": -0.0,
    "J24_ELBOW_YAW_R": 0.0,
    "J27_HEAD_PITCH": 0.0,
    "J28_HEAD_YAW": 0.0,
}