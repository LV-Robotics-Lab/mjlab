# ROS2 rl_dance.yaml 与 mjlab 训练参数对比

## 1. 观测配置 (Observations)

### ROS2 (rl_dance.yaml)
- **观测类型**: `mimic_future` - 返回当前+未来10帧的参考关节位置和速度
- **goal_length**: 480 (current(24*2=48) + future_9_frames(9*24*2=432))
- **goal_history_steps**: 1
- **num_include_obs_steps**: 5 (本体感知历史步数)
- **goal_scale**: 1.0
- **num_single_observations**: 78

**观测缩放 (observation_scale)**:
- `observation_scale_linear_vel`: 1.0
- `observation_scale_angular_vel`: 1.0
- `observation_scale_dof_pos`: 1.0
- `observation_scale_dof_vel`: 0.05 (关节速度缩放为1/20) ⚠️ **关键：goal数据中的joint_vel部分会被应用此缩放**
- `observation_scale_quat`: 1.0
- `observation_scale_actions`: 1.0
- `observation_scale_command`: 1.0 (整体command不缩放，但内部vel部分会应用observation_scale_dof_vel)
- `observation_scale_future_frames`: 1.0 (整体future_frames不缩放，但内部vel部分会应用observation_scale_dof_vel)

**⚠️ 重要发现**：
- ROS2 的 `mimic_future` goal 数据格式：`[joint_pos, joint_vel]` (当前帧) + 9帧 `[joint_pos, joint_vel]`
- 虽然 `observation_scale_command: 1.0` 和 `observation_scale_future_frames: 1.0`，但 goal 数据内部的 `joint_vel` 部分会应用 `observation_scale_dof_vel: 0.05`
- 这意味着训练出的 MNN 模型期望接收的 goal 数据中，所有 `joint_vel` 值都被缩放了 0.05

### mjlab (基于实际代码: `src/mjlab/tasks/tracking/`)
- **观测类型**: 包含多个观测项，每个都有历史信息
  - `joint_pos`: 相对关节位置（相对于默认位置），history_length=5，噪声范围[-0.01, 0.01]
  - `joint_vel`: 相对关节速度（相对于默认速度），history_length=5，噪声范围[-0.5, 0.5]
  - `actions`: 上一步动作，history_length=5
  - `base_ang_vel`: 基座角速度，history_length=5，噪声范围[-0.2, 0.2]
  - `projected_gravity`: 投影重力向量（在机器人锚点坐标系中），history_length=5
  - `motion_anchor_ori_b`: 锚点方向误差（参考与机器人的方向差），history_length=5，噪声范围[-0.05, 0.05]
  - `command`: 当前帧的关节位置和速度目标 `[joint_pos, joint_vel]`，shape=(N, num_joints*2)
  - `future_frames`: 未来9帧的关节位置和速度目标，按帧顺序堆叠，shape=(N, 9*num_joints*2)

**观测归一化**:
- `actor_obs_normalization`: True
- `critic_obs_normalization`: True

**关键实现细节** (来自 `commands.py` 和 `observations.py`):
- `command.command` 返回: `torch.cat([joint_pos * pos_scale, joint_vel * vel_scale], dim=1)` - 当前帧，shape=(N, num_joints*2)
- `future_frames_command` 返回: 未来9帧（t+1到t+9），每帧包含 `[pos * pos_scale, vel * vel_scale]`，按帧顺序堆叠，shape=(N, 9*num_joints*2)
- **当前配置**：使用 `generated_commands_with_scale` 和 `future_frames_generated_commands_with_scale`，设置 `pos_scale=1.0, vel_scale=0.05` ✅ 与 ROS2 一致
- **解决方案**：mjlab 应该使用 `generated_commands_with_scale` 和 `future_frames_generated_commands_with_scale`，设置 `vel_scale=0.05` 来匹配 ROS2 的行为
- 历史观测通过 `history_length=5` 和 `flatten_history_dim=True` 实现，将5帧历史展平
- 对于PM1机器人（24个关节），观测维度估算：
  - `joint_pos`: 24 * 5 = 120
  - `joint_vel`: 24 * 5 = 120
  - `actions`: 24 * 5 = 120
  - `base_ang_vel`: 3 * 5 = 15
  - `projected_gravity`: 3 * 5 = 15
  - `motion_anchor_ori_b`: 6 * 5 = 30 (旋转矩阵前两列)
  - `command`: 24 * 2 = 48 (当前帧)
  - `future_frames`: 9 * 24 * 2 = 432 (未来9帧)
  - 总计约: 900+ 维（不含其他可能的观测项）

## 2. 动作配置 (Actions)

### ROS2 (rl_dance.yaml)
- **action_scale**: 每个关节不同（基于effort_limit和stiffness计算）
  - 左腿: [0.229, 0.229, 0.491, 0.229, 0.418, 0.418]
  - 右腿: [0.229, 0.229, 0.491, 0.229, 0.418, 0.418]
  - 腰部: [0.491]
  - 左臂: [0.491, 0.491, 0.491, 0.491, 0.491]
  - 右臂: [0.491, 0.491, 0.491, 0.491, 0.491]
  - 头部: [0.491]
- **action_clip**: 1000.0
- **qd_mask**: 关节速度掩码（踝关节不使用速度控制）
  - 左腿/右腿: [1.0, 1.0, 1.0, 1.0, 0, 0]
  - 其他关节: 全为1.0

### mjlab
- **动作类型**: `JointPositionActionCfg`
- **action_scale**: `PM_ACTION_SCALE` (动态计算: `0.25 * effort_limit / stiffness`，ANKLE 关节额外乘以 0.85)
  - Q90 关节（HIP_PITCH, HIP_ROLL, KNEE_PITCH）: **0.229** ✅ 与 ROS2 yaml 一致
  - Q25 关节（HIP_YAW, WAIST, ARMS, HEAD）: **0.491** ✅ 与 ROS2 yaml 一致
  - Q25 关节（ANKLE_PITCH, ANKLE_ROLL）: **0.418** ✅ 与 ROS2 yaml 一致
- **clip_actions**: 1000.0 ✅ 与 ROS2 action_clip 一致

## 3. 控制参数 (Control)

### ROS2 (rl_dance.yaml)
- **joint_kp**: PD控制器的位置增益
- **joint_kd**: PD控制器的速度增益
- **default_joint_q**: 默认关节位置

### mjlab
- **控制方式**: 通过MuJoCo的BuiltinPositionActuator实现
- **stiffness/damping**: 对应 ROS2 的 kp/kd，值完全一致 ✅
- **default_joint_pos**: 对应 ROS2 的 default_joint_q，值完全一致 ✅

## 4. 扭矩限制 (Torque Limits)

### ROS2 (rl_dance.yaml)
- **torque_limit**: True
- **max_torque_joint**: 每个关节的扭矩限制
- **max_lower_body_torque**: 550.0

### mjlab
- **扭矩限制**: 通过MuJoCo的effort_limit在机器人配置中定义
- 所有关节的effort_limit与ROS2的max_torque_joint完全一致 ✅


## 7. 仿真参数 (Simulation)

### ROS2 (rl_dance.yaml)
- **periods**: 0.02 (50Hz控制频率)
- **transition_time**: 0.01 (初始启动时的平滑过渡时间)

### mjlab
- **timestep**: 0.005, **decimation**: 4 (实际控制频率: 50Hz) ✅ 与 ROS2 一致
- **iterations**: 10
- **ls_iterations**: 20
- **episode_length_s**: 10.0
- **num_envs**: 4096 (训练时)
- **transition_time**: ⚠️ **部分等效**（mjlab 通过 PD 控制器自然过渡，但过渡时间取决于误差大小和系统动力学，可能无法精确控制在 0.01 秒内）
  - PD 参数：自然频率 10Hz，阻尼比 2.0（过阻尼）
  - 理论过渡时间（settling time）≈ 0.032 秒（误差<2%），但实际时间取决于初始误差
  - ROS2 的 transition_time 是明确的时间限制（0.01秒），而 mjlab 的过渡时间是动态的
- **observation_clip**: ✅ 已添加（所有观测项都设置了 clip=(-20000.0, 20000.0)，与 ROS2 的 observation_clip: 20000.0 一致）

## 观测 Scale 问题及解决方案

### 问题描述

同一个 MNN 模型在 ROS2 和 mjlab 中表现不同，原因是**观测 scale 不匹配**。

### 解决方案 ✅ 已实施

**mjlab 已更新为使用 scaled 版本的 command**：
- 在 `tracking_env_cfg.py` 中使用 `generated_commands_with_scale` 和 `future_frames_generated_commands_with_scale`
- 设置 `pos_scale=1.0, vel_scale=0.05` 以匹配 ROS2 的 `observation_scale_dof_vel: 0.05`
- 这样训练出的模型在 ROS2 和 mjlab 中应该表现一致

**如果模型是在 mjlab 中训练的**（使用 `vel_scale=0.05`）：
- ✅ **ROS2 应该使用 `observation_scale_dof_vel: 0.05`**（与 mjlab 训练时一致）
- mjlab 训练时：vel 值被缩放了 0.05
- 归一化统计基于缩放过后的分布
- ROS2 推理时也应该使用相同的缩放来匹配训练时的分布

**如果模型是在 ROS2 中训练的**（使用 `observation_scale_dof_vel: 0.05`）：
- ✅ **mjlab 已配置为使用 `generated_commands_with_scale(vel_scale=0.05)`**
- ROS2 训练时：vel 值被缩放了 0.05
- mjlab 推理时使用相同的缩放，观测分布一致

### 归一化机制说明

- `actor_obs_normalization=True` 会在训练时计算观测的均值和方差，然后归一化
- 推理时使用训练时保存的统计进行归一化
- **关键**：归一化统计是基于训练时的观测分布，所以推理时的观测分布必须与训练时一致

## 主要差异总结

1. **观测系统**:
   - **ROS2**: 使用`mimic_future`类型，包含当前帧+未来9帧数据（goal_length=480），共10帧
   - **mjlab**: 使用历史5帧数据 + 当前帧命令 + 未来9帧命令，包含更多观测项
     - 历史观测：joint_pos, joint_vel, actions, base_ang_vel, projected_gravity, motion_anchor_ori_b (每个都有5帧历史)
     - 当前帧命令：`[joint_pos * 1.0, joint_vel * 0.05]` (使用 scaled 版本，vel_scale=0.05)
     - 未来9帧命令：按帧顺序堆叠的 `[pos * 1.0, vel * 0.05]` 数据 (使用 scaled 版本，vel_scale=0.05)

2. **观测缩放** ✅ **已解决**:
   - **ROS2**: 
     - goal 数据（command + future_frames）中的 `joint_vel` 部分被缩放了 0.05
     - 训练出的 MNN 模型期望接收缩放过后的 vel 数据
   - **mjlab**: 
     - ✅ 已更新为使用 `generated_commands_with_scale(vel_scale=0.05)` 和 `future_frames_generated_commands_with_scale(vel_scale=0.05)`
     - 观测归一化通过 `actor_obs_normalization=True` 实现
     - **结果**：mjlab 和 ROS2 的观测分布一致，模型兼容性已解决

3. **动作缩放**:
   - **ROS2**: 使用固定的action_scale数组（每个关节不同）
   - **mjlab**: 动态计算action_scale (0.25 * effort_limit / stiffness)

4. **控制方式**:
   - **ROS2**: 显式定义kp/kd参数（PD控制器）
   - **mjlab**: 通过MuJoCo的BuiltinPositionActuator配置隐式控制（stiffness和damping）

5. **奖励函数**:
   - **ROS2**: 配置文件中未显示（可能在C++代码中定义）
   - **mjlab**: 有详细的奖励函数配置和权重（在 `tracking_env_cfg.py` 中定义）

6. **RL算法**:
   - **ROS2**: 配置文件中未显示
   - **mjlab**: 使用PPO算法，有完整的超参数配置（在 `rl_cfg.py` 中定义）

7. **仿真频率**:
   - **ROS2**: 50Hz (csv_dt=0.02)
   - **mjlab**: 50Hz控制频率 (timestep=0.005, decimation=4)

8. **训练规模**:
   - **ROS2**: 未显示
   - **mjlab**: 4096个并行环境，最大迭代30,000次

9. **未来帧处理**:
   - **ROS2**: 未来帧数据包含在goal中，goal_length=480（当前24*2 + 未来9帧*24*2）
   - **mjlab**: 未来帧通过 `future_frames_generated_commands` 单独提供，shape=(N, 9*num_joints*2)
