# 使用 amp-rsl-rl 做 AMP 训练

本项目使用 [amp-rsl-rl](https://github.com/ami-iit/amp-rsl-rl) 进行 AMP（Adversarial Motion Priors）训练。环境侧提供 `disc_obs`、`get_disc_obs_space()`、`fetch_disc_obs_demo()`，训练时使用 amp-rsl-rl 的 runner 与算法。

## 安装

```bash
pip install amp-rsl-rl
# 或使用本项目可选依赖
uv pip install -e ".[amp]"   # 或 pip install -e ".[amp]"
```

安装后，任务列表会出现 **`Mjlab-Falling-Flat-PM1-AMP`**；未安装时该任务不会注册。

## 环境端接口

- **`ManagerBasedRlEnvCfg.amp`**：`AMPCfg(motion_file=...)`（单文件或 `list[str]` 多个 `.npz`）。
- **step() 的 extras**：当前 disc 观测同时放在 **`extras["disc_obs"]`** 和 **`extras["amp_obs"]`**（别名），兼容 amp-rsl-rl 可能使用的 key；reset() 若带 `disc_obs` 也会同步 `amp_obs`。
- **`get_disc_obs_space()`**：在 wrapper 或 unwrapped 环境上调用，返回 disc 观测的 `Box`。
- **`fetch_disc_obs_demo(num_samples)`**：在 wrapper 或 unwrapped 环境上调用，返回参考 disc 观测（来自本仓库 .npz 或站立合成）。

Fall 任务在 `env_cfg` 中设置 `amp=AMPCfg(...)`；PM1 的 `env_cfgs.py` 中可通过 `cfg.amp.motion_file` 指定参考动作 `.npz`。

## Demo 数据格式

本仓库使用 **.npz**（`joint_pos`, `joint_vel`，可选 `root_pos` / `root_quat` / `root_lin_vel` / `root_ang_vel` 或 `body_*`），由 `fetch_disc_obs_demo()` 从 env 侧采样，**不**走 amp-rsl-rl 文档里的 .npy 数据集加载。若要用 amp-rsl-rl 的 .npy 格式（如 `joint_positions`, `root_position`, `root_quaternion` 等），需自行写脚本将 .npz 转成其要求的结构或转成我们 .npz 的 key。

## Runner 与 config

- **AMP 任务**（`Mjlab-Falling-Flat-PM1-AMP`）使用 **amp-rsl-rl 的 OnPolicyRunner**（`amp_rsl_rl.runners.OnPolicyRunner`），不是 rsl_rl 的 OnPolicyRunner；仅在安装 amp-rsl-rl 后该任务才会注册。
- 算法为 amp-rsl-rl 的 AMP PPO（判别器、reward 混合、replay 等），超参在 `pm1_falling_amp_runner_cfg()` 里（disc 学习率、reward 权重、disc hidden 等）。

## 训练

安装 amp-rsl-rl 后，用本仓库统一训练入口跑 AMP 任务：

```bash
python -m mjlab.scripts.train Mjlab-Falling-Flat-PM1-AMP
```

算法配置在 `mjlab.tasks.fall.config.pm1.rl_cfg.pm1_falling_amp_runner_cfg()`，其中 `algorithm.class_name` 指向 `amp_rsl_rl.algorithms.amp_ppo.AMP_PPO`。若你使用的 amp-rsl-rl 版本中算法类路径不同，请在 `RslRlAmpAlgorithmCfg` 中修改 `class_name`。
