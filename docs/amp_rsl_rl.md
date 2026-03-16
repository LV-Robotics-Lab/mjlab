# 使用 amp-rsl-rl 做 AMP 训练

本项目已移除自实现的 AMP runner（`AMP_PPO`），改用 [amp-rsl-rl](https://github.com/ami-iit/amp-rsl-rl) 进行 AMP（Adversarial Motion Priors）训练，以避免与 rsl-rl 对接时的梯度问题。

## 环境端 AMP 接口（保留）

以下能力仍在 **环境侧** 提供，供 amp-rsl-rl 使用：

- **`ManagerBasedRlEnvCfg.amp`**：可配置 `AMPCfg(motion_file=...)`（单文件或 `list[str]` 多个 `.npz`）。
- **`extras["disc_obs"]`**：每步在 `extras` 中提供当前 disc 观测。
- **`get_disc_obs_space()`**：在 unwrapped 环境上调用，返回 disc 观测的 `gym.spaces.Box`。
- **`fetch_disc_obs_demo()`**：在 unwrapped 环境上调用，返回参考动作的 disc 观测，用于判别器训练。

Fall 任务中可在 `env_cfg` 里设置 `amp=AMPCfg(...)`，PM1 的 `env_cfgs.py` 中可通过 `cfg.amp.motion_file` 指定参考动作文件。

## 使用 amp-rsl-rl

1. **安装依赖**：`pip install amp-rsl-rl`（或按 amp-rsl-rl 仓库说明安装）。
2. **使用 amp-rsl-rl 的 runner/算法**：用其提供的 AMP 版 PPO runner 和算法类进行训练，不再使用本仓库已删除的 `AMP_PPO`。
3. **对接本仓库环境**：确保 amp-rsl-rl 使用的 env 接口与上面一致（`extras["disc_obs"]`、`get_disc_obs_space()`、`fetch_disc_obs_demo()`）。若 amp-rsl-rl 使用不同 key 或方法名，可在 `RslRlVecEnvWrapper` 外再包一层薄 wrapper，将上述接口映射过去。
4. **Fall + AMP**：选择带 `amp=AMPCfg(...)` 的 fall env_cfg（例如 PM1 的 `pm1_flat_falling_env_cfg()`，并在对应 env_cfg 中设置 `amp.motion_file`），用 amp-rsl-rl 的脚本或入口启动训练即可。
