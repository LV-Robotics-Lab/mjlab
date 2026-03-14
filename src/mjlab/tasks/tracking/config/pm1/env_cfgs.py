"""PM1 flat tracking environment configurations."""

import copy
from pathlib import Path

from mjlab.asset_zoo.robots import (
  PM_ACTION_SCALE,
  PM_ROBOT_CFG,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import reset_root_state_fall_velocity
from mjlab.managers.manager_term_config import (
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.rewards import survival_bonus
from mjlab.tasks.tracking.mdp.observations import (
  episode_fall_direction,
  future_frames_generated_commands_with_scale_selected,
  generated_commands_with_scale_selected,
  motion_anchor_ori_b_selected,
  projected_gravity_selected,
)
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# 护具 map 数据目录（默认站立系下的查表文件放于此，reward 中力衰减可引用）
PROTECTOR_MAP_DIR = Path(__file__).resolve().parent / "protector_map"


def pm1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
  use_protector_map: bool = True,
) -> ManagerBasedRlEnvCfg:
  """创建 PM1 平地跟踪任务配置。
  
  Args:
    has_state_estimation: 如果为 True，包含 base_lin_vel 和 motion_anchor_pos_b 观测。
                          如果为 False，移除这些观测（用于没有状态估计的系统）。
    play: 如果为 True，配置为播放模式（无限episode，无噪声，无随机化）。
    use_protector_map: 如果为 True，reduce_contact_force 使用护具 map + 力衰减公式；
                       为 False 时直接用传感器力做加权惩罚，不使用护具查表。
  
  Returns:
    ManagerBasedRlEnvCfg: 配置好的 PM1 跟踪任务环境。
  """
  cfg = make_tracking_env_cfg()

  # 设置 PM1 机器人配置
  cfg.scene.entities = {"robot": PM_ROBOT_CFG}

  ##
  # 接触传感器配置
  ##
  
  # PM1 机器人自碰撞检测
  # 检测机器人各身体部件之间的碰撞
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  # 检测所有机器人身体部件与地面的最大接触力
  # 排除 ankle_pitch 和 ankle_roll（脚部预期会接触地面）
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
    fields=("force", "found", "pos"),
    reduce="maxforce",
    num_slots=1,
  )
  cfg.scene.sensors = (self_collision_cfg, body_contact_force_cfg,)

  ##
  # 动作配置
  ##
  
  # 根据 PM1 机器人的扭矩限制和刚度设置每个关节的动作缩放
  # PM_ACTION_SCALE 计算公式为：0.25 * effort_limit / stiffness（每个关节）
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PM_ACTION_SCALE

  ##
  # 运动命令配置
  ##
  
  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  # 设置锚点身体为基座链接，用于运动跟踪
  motion_cmd.anchor_body_name = "LINK_BASE"
  # 定义运动命令中要跟踪的身体链接
  # 注释掉的链接不用于跟踪（例如，某些 pitch/yaw 关节）
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

  ##
  # 域随机化事件
  ##
  
  # 训练时随机化脚部摩擦系数
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^collision_(left|right)_foot(_toe)?$"
  # 训练时随机化基座质心位置
  cfg.events["base_com"].params["asset_cfg"].body_names = ("LINK_BASE",)

  ##
  # 终止条件
  ##
  
  # 如果末端执行器（踝关节）位置偏离目标太远，终止 episode
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R",
  )
  # 头部冲击过大时终止（避免手撑地后头部轻微贴地被误杀）
  cfg.terminations["forbidden_body_contact_force"].params["body_names"] = ("LINK_HEAD_YAW", "LINK_TORSO_YAW", "LINK_ELBOW_END_L", "LINK_ELBOW_END_R")
  cfg.terminations["forbidden_body_contact_force"].params["force_threshold"] = 20000.0

  ##
  # 奖励：reduce_contact_force 可选护具 map + 力衰减公式
  ##
  cfg.rewards["reduce_contact_force"].params["asset_cfg"] = SceneEntityCfg("robot")
  if use_protector_map:
    cfg.rewards["reduce_contact_force"].params["protector_map_dir"] = PROTECTOR_MAP_DIR
    cfg.rewards["reduce_contact_force"].params["force_params_path"] = PROTECTOR_MAP_DIR / "fitted_parameters.json"
    cfg.rewards["reduce_contact_force"].params["density"] = 0.3
  else:
    cfg.rewards["reduce_contact_force"].params["protector_map_dir"] = None
    cfg.rewards["reduce_contact_force"].params["force_params_path"] = None
  ##
  # 查看器配置
  ##
  
  # 设置查看器跟随基座链接
  cfg.viewer.body_name = "LINK_BASE"

  ##
  # PM1 机器人传感器名称修复
  ##
  
  # 修复 PM1 机器人的传感器名称（与 G1 使用不同的传感器名称）
  # PM1 使用: imu_link_linear_velocity, imu_angular_velocity
  # G1 使用: imu_lin_vel, imu_ang_vel
  if "base_lin_vel" in cfg.observations["policy"].terms:
    cfg.observations["policy"].terms["base_lin_vel"].params["sensor_name"] = "robot/imu_link_linear_velocity"
  if "base_ang_vel" in cfg.observations["policy"].terms:
    cfg.observations["policy"].terms["base_ang_vel"].params["sensor_name"] = "robot/imu_angular_velocity"
  if "base_lin_vel" in cfg.observations["critic"].terms:
    cfg.observations["critic"].terms["base_lin_vel"].params["sensor_name"] = "robot/imu_link_linear_velocity"
  if "base_ang_vel" in cfg.observations["critic"].terms:
    cfg.observations["critic"].terms["base_ang_vel"].params["sensor_name"] = "robot/imu_angular_velocity"

  ##
  # 观测修改
  ##
  
  # 如果没有状态估计，修改观测
  # 移除需要状态估计的 base_lin_vel 和 motion_anchor_pos_b
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

  ##
  # 播放模式配置
  ##
  
  # 应用播放模式覆盖设置（用于推理/评估）
  if play:
    # 无限 episode 长度（无自动终止）
    cfg.episode_length_s = int(1e9)

    # 禁用观测噪声/损坏，用于干净的推理
    cfg.observations["policy"].enable_corruption = False
    # 播放时禁用随机推动与 reset 初速度
    cfg.events.pop("push_robot", None)
    cfg.events.pop("reset_base_velocity", None)

    # 禁用 RSI（随机状态初始化）随机化
    # episode 开始时无随机姿态/速度变化
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    # 始终从运动文件的开头开始
    motion_cmd.sampling_mode = "start"

  return cfg

def pm1_distill_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
  motion_forward_file: str = "",
  motion_backward_file: str = "",
  use_protector_map: bool = True,
) -> ManagerBasedRlEnvCfg:
  """PM1 蒸馏任务环境配置：Student 不做 tracking，Teacher 用双 motion 观测。

  - Student：policy/critic 观测无 motion。
  - Teacher：观测与 tracking critic 一致，按 fall_direction 选 motion_forward / motion_backward。
  - 双 motion 文件可在本函数参数提前指定，或训练时用 --motion-forward-file / --motion-backward-file 覆盖。
  - use_protector_map: 为 True 时 reduce_contact_force 使用护具 map；为 False 时不用护具查表。
  """
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": PM_ROBOT_CFG}

  ## 接触传感器
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
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
    fields=("force", "found", "pos"),
    reduce="maxforce",
    num_slots=1,
  )
  cfg.scene.sensors = (self_collision_cfg, body_contact_force_cfg,)

  ## 动作
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PM_ACTION_SCALE

  ## 双 motion 命令：前摔/后摔各一个，仅供 Teacher 观测；训练时需 --motion-forward-file 与 --motion-backward-file
  assert cfg.commands is not None and "motion" in cfg.commands
  motion_template = copy.deepcopy(cfg.commands["motion"])
  assert isinstance(motion_template, MotionCommandCfg)
  motion_template.anchor_body_name = "LINK_BASE"
  motion_template.body_names = (
    "LINK_BASE",
    "LINK_HIP_ROLL_L",
    "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    "LINK_HIP_ROLL_R",
    "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    "LINK_TORSO_YAW",
    "LINK_SHOULDER_ROLL_L",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_YAW_L",
    "LINK_SHOULDER_ROLL_R",
    "LINK_ELBOW_PITCH_R",
    "LINK_ELBOW_YAW_R",
    "LINK_HEAD_YAW",
  )
  motion_template.motion_file = motion_forward_file or ""
  cfg.commands["motion_forward"] = copy.deepcopy(motion_template)
  cfg.commands["motion_backward"] = copy.deepcopy(motion_template)
  cfg.commands["motion_backward"].motion_file = motion_backward_file or ""
  del cfg.commands["motion"]

  ## Teacher 观测：与训练 Teacher 时一致，即 tracking 的 policy 观测（约 900 维），motion 相关项按 fall_direction 选 motion_forward / motion_backward
  _cmd_fwd, _cmd_bwd = "motion_forward", "motion_backward"
  _scale = {"pos_scale": 1.0, "vel_scale": 0.05}
  teacher_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      history_length=5,
      flatten_history_dim=True,
      clip=(-20000.0, 20000.0),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      history_length=5,
      flatten_history_dim=True,
      clip=(-20000.0, 20000.0),
    ),
    "actions": ObservationTermCfg(
      func=mdp.last_action,
      history_length=5,
      flatten_history_dim=True,
      clip=(-20000.0, 20000.0),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_angular_velocity"},
      history_length=5,
      flatten_history_dim=True,
      clip=(-20000.0, 20000.0),
    ),
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity_selected,
      params={"command_forward": _cmd_fwd, "command_backward": _cmd_bwd},
      history_length=5,
      flatten_history_dim=True,
      clip=(-20000.0, 20000.0),
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=motion_anchor_ori_b_selected,
      params={"command_forward": _cmd_fwd, "command_backward": _cmd_bwd},
      history_length=5,
      flatten_history_dim=True,
      clip=(-20000.0, 20000.0),
    ),
    "command": ObservationTermCfg(
      func=generated_commands_with_scale_selected,
      params={"command_forward": _cmd_fwd, "command_backward": _cmd_bwd, **_scale},
      clip=(-20000.0, 20000.0),
    ),
    "future_frames": ObservationTermCfg(
      func=future_frames_generated_commands_with_scale_selected,
      params={"command_forward": _cmd_fwd, "command_backward": _cmd_bwd, **_scale},
      clip=(-20000.0, 20000.0),
    ),
  }
  cfg.observations["teacher"] = ObservationGroupCfg(
    terms=teacher_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )

  ## 摔倒防护：reset 时施加前向或后向初速度，并设置 episode_fall_direction 供双 Teacher 选择
  cfg.events.pop("reset_base_velocity", None)
  cfg.events["reset_fall_velocity"] = EventTermCfg(
    func=reset_root_state_fall_velocity,
    mode="reset",
    params={
      "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
      "speed_range": (0.5, 1.5),
      "forward_prob": 0.5,
    },
  )

  ## 域随机化
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^collision_(left|right)_foot(_toe)?$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("LINK_BASE",)

  ## 终止条件：删除依赖 motion 的项，仅保留 time_out 与 forbidden_body_contact_force
  cfg.terminations.pop("anchor_pos", None)
  cfg.terminations.pop("anchor_ori", None)
  cfg.terminations.pop("ee_body_pos", None)
  cfg.terminations["forbidden_body_contact_force"].params["body_names"] = (
    "LINK_HEAD_YAW",
    # "LINK_TORSO_YAW",
    # "LINK_ELBOW_END_L",
    # "LINK_ELBOW_END_R",
  )
  cfg.terminations["forbidden_body_contact_force"].params["force_threshold"] = 1000.0

  ## 奖励：删除所有与 motion 相关的 tracking 奖励
  motion_reward_keys = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "feet_relative_pos",
    "projected_gravity_tracking",
    "ankle_pitch_joint_tracking",
    "ankle_roll_joint_tracking",
    "ankle_joint_smoothness",
    "ankle_joint_jerk_penalty",
  )
  for k in motion_reward_keys:
    cfg.rewards.pop(k, None)
  # 每步小正奖励，给策略「存活」方向，避免纯惩罚导致早停优化
  cfg.rewards["survival_bonus"] = RewardTermCfg(func=survival_bonus, weight=0.01)
  cfg.rewards["reduce_contact_force"].params["asset_cfg"] = SceneEntityCfg("robot")
  if use_protector_map:
    cfg.rewards["reduce_contact_force"].params["protector_map_dir"] = PROTECTOR_MAP_DIR
    cfg.rewards["reduce_contact_force"].params["force_params_path"] = PROTECTOR_MAP_DIR / "fitted_parameters.json"
    cfg.rewards["reduce_contact_force"].params["density"] = 0.3
  else:
    cfg.rewards["reduce_contact_force"].params["protector_map_dir"] = None
    cfg.rewards["reduce_contact_force"].params["force_params_path"] = None
  cfg.rewards["reduce_contact_force"].weight = 0.0001
  cfg.rewards["action_rate_l2"].weight = -0.001

  cfg.viewer.body_name = "LINK_BASE"

  ## 观测：删除与 motion 相关项。Policy 仅保留 joint_pos, joint_vel, actions, base_ang_vel
  policy_drop = {"command", "future_frames", "projected_gravity", "motion_anchor_ori_b", "motion_anchor_pos_b", "base_lin_vel"}
  new_policy_terms = {
    k: v for k, v in cfg.observations["policy"].terms.items()
    if k not in policy_drop
  }
  if not has_state_estimation:
    new_policy_terms.pop("base_lin_vel", None)
  if "base_ang_vel" in new_policy_terms:
    new_policy_terms["base_ang_vel"].params["sensor_name"] = "robot/imu_angular_velocity"
  cfg.observations["policy"] = ObservationGroupCfg(
    terms=new_policy_terms,
    concatenate_terms=True,
    enable_corruption=True,
  )

  ## Critic 观测：删除 motion 相关项，保留 base_ang_vel, joint_pos, joint_vel, actions, base_lin_vel
  critic_drop = {
    "command", "future_frames", "projected_gravity_error",
    "motion_anchor_ori_b", "motion_anchor_pos_b", "body_pos", "body_ori",
  }
  new_critic_terms = {
    k: v for k, v in cfg.observations["critic"].terms.items()
    if k not in critic_drop
  }
  if not has_state_estimation:
    new_critic_terms.pop("base_lin_vel", None)
  if "base_lin_vel" in new_critic_terms:
    new_critic_terms["base_lin_vel"].params["sensor_name"] = "robot/imu_link_linear_velocity"
  if "base_ang_vel" in new_critic_terms:
    new_critic_terms["base_ang_vel"].params["sensor_name"] = "robot/imu_angular_velocity"
  new_critic_terms["fall_direction"] = ObservationTermCfg(func=episode_fall_direction)
  cfg.observations["critic"] = ObservationGroupCfg(
    terms=new_critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )

  ## 删除 curriculum（依赖 motion 的 initial_velocity 等）
  cfg.curriculum = {}

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("reset_fall_velocity", None)

  return cfg