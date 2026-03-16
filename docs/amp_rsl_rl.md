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
- **`extras["disc_obs"]`**：每步在 `extras` 中提供当前 disc 观测。
- **`get_disc_obs_space()`**：在 wrapper 或 unwrapped 环境上调用，返回 disc 观测的 `Box`。
- **`fetch_disc_obs_demo(num_samples)`**：在 wrapper 或 unwrapped 环境上调用，返回参考 disc 观测。

Fall 任务在 `env_cfg` 中设置 `amp=AMPCfg(...)`；PM1 的 `env_cfgs.py` 中可通过 `cfg.amp.motion_file` 指定参考动作 `.npz`。

## 训练

安装 amp-rsl-rl 后，用本仓库统一训练入口跑 AMP 任务：

```bash
python -m mjlab.scripts.train Mjlab-Falling-Flat-PM1-AMP
```

算法配置在 `mjlab.tasks.fall.config.pm1.rl_cfg.pm1_falling_amp_runner_cfg()`，其中 `algorithm.class_name` 指向 `amp_rsl_rl.algorithms.amp_ppo.AMP_PPO`。若你使用的 amp-rsl-rl 版本中算法类路径不同，请在 `RslRlAmpAlgorithmCfg` 中修改 `class_name`。
