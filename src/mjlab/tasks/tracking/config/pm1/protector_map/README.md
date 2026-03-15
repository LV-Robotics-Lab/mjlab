# 护具 map（Protector Map）

本目录存放**默认站立系**下的护具/防护查表数据，供 tracking reward 中力衰减使用。

- **坐标系**：与 `contact_pos_in_default_standing_frame()` 输出一致（仿真**世界系**，机器人处于 default qpos 时的世界系）。
- **用途**：用接触点默认系坐标 `pos_default [B, N, 3]` 查表得到厚度/衰减，再乘到接触力上。

### 原点 (0, 0, 0)

- **(0, 0, 0)** = **世界原点**（仿真世界系原点）。平地场景下 **z = 0 即地面高度**。
- 因此 map 的 **z ∈ [0, 1.4]** 表示**离地高度** 0～1.4 m；**y ∈ [-0.4, 0.4]** 为左右；**x** 正为前、负为后（与 default 站立时机器人前方一致）。
- PM 默认站立时 root 约在 (0, 0, 0.76)，即脚踝离地约 0.76 m；map 覆盖从地面到约 1.4 m 高。

## YZ 平面 TSV 网格

| 文件 | 适用区域 | 说明 |
|------|----------|------|
| `yz_map_front.tsv` | x ≥ 0（前侧） | YZ 平面网格 |
| `yz_map_back.tsv` | x < 0（后侧） | YZ 平面网格 |

- **Y**：-0.4 m ~ +0.4 m，步长 0.05 m（17 列）
- **Z**：0 m ~ 1.4 m，步长 0.05 m（29 行）
- **像素**：0.05 m × 0.05 m 方格
- **像素值**：护具厚度（mm），如 0、6、12、18、24
- **全局密度**：首段注释后单独一行 `# density 0.4`（无量纲，全表共用一个 p，进公式 `p^β`）。不写则沿用 reward 参数 `density`。

**查表逻辑**：表头/行首的 Y、Z 是**格心**坐标。读 map 时根据格心自动算间距 dy, dz = (max−min)/(n−1)，只算一次并缓存在仿真中复用；最近邻查表用该 dy, dz，故每格覆盖以格心为中心、边长 (dy, dz) 的区域（如 0.05 m 步长则 ±0.025 m）。  
例如 **Y=0 列、Z=0.1 行** 的值为 6 → 表示该格心 (0, 0.1) 对应区域（约 **-0.025 ≤ y < 0.025、0.075 ≤ z < 0.125** m）厚度 6 mm。

TSV 格式：首行为注释，第二行为表头 `z\y` + Y 坐标，以下每行为 Z 值 + 该行各格像素值（制表符分隔）。

## 力衰减参数

`fitted_parameters.json`：与 `scripts/ThicknessCalculate/force_calculator.py` 同公式的拟合参数（C, alpha, beta, gamma）。  
reward 中按 `F_after = C * (t_mm^alpha) * (p^beta) * (F_before^gamma)` 计算衰减后力（**t** 按格查表，**p** 为 TSV 里一行 `# density`；F 单位 kN）。

## 路径引用

```python
from mjlab.tasks.tracking.config.pm1.env_cfgs import PROTECTOR_MAP_DIR

path_front = PROTECTOR_MAP_DIR / "yz_map_front.tsv"
path_back = PROTECTOR_MAP_DIR / "yz_map_back.tsv"
```
