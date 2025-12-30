训练
python -m mjlab.scripts.train Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Back_1_converted.npz --env.scene.num-envs 4096 --agent.max_iterations 10000

# 恢复训练 - 从 WandB 恢复（推荐）
python -m mjlab.scripts.train Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/motion.npz \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 10000 \
  --agent.resume True \
  --wandb-run-path e1519767-national-university-of-singapore/mjlab/run-id

# 恢复训练 - 从本地文件系统恢复
python -m mjlab.scripts.train Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/motion.npz \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 10000 \
  --agent.resume True \
  --agent.load-run "2025-12-14_17-37-01" \
  --agent.load-checkpoint "model_1000.pt"

演示
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --wandb-run-path e1519767-national-university-of-singapore/mjlab/bo5t1dmw
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/motion.npz --wandb-run-path 1205492990-nus/mjlab/vboc51sb
python -m mjlab.scripts.force Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Forward_1_converted.npz --wandb-run-path 1205492990-nus/mjlab/vboc51sb

纯mimic向前摔：1205492990-nus/mjlab/f9mbpspg
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Front_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/f9mbpspg

纯mimic向后摔：1205492990-nus/mjlab/h0qf16ob
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Back_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/h0qf16ob

纯mimic向左摔：1205492990-nus/mjlab/obw2ysrf
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Left_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/obw2ysrf

纯mimic向右摔：1205492990-nus/mjlab/gc5ovv94
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Right_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/gc5ovv94

纯mimic（从左前）向右后摔：1205492990-nus/mjlab/nebt84gj
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/LeftFront_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/nebt84gj

纯mimic（从左后）向右前摔：1205492990-nus/mjlab/x2ohjmc5
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/LeftBack_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/x2ohjmc5

纯mimic（从右前）向左后摔：1205492990-nus/mjlab/mk5mwfqe
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/RightFront_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/mk5mwfqe

纯mimic（从右后）向左前摔：1205492990-nus/mjlab/l7f3pf0x
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/RightBack_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/l7f3pf0x

--wandb-run-path 1205492990-nus/mjlab/6icim82d

## MNN 模型转换

### 安装 MNN

MNN 转换工具通常已安装在系统中（`/usr/local/bin/MNNConvert`）。如果未安装，可以从源码编译：

```bash
# 克隆 MNN 仓库
git clone https://github.com/alibaba/MNN.git
cd MNN

# 编译（需要 CMake）
mkdir build && cd build
cmake .. -DMNN_BUILD_CONVERTER=ON
make -j4

# 编译完成后，MNNConvert 位于 build/MNNConvert
# 可以复制到系统路径或添加到 PATH
sudo cp MNNConvert /usr/local/bin/
```

或者如果系统已安装，直接使用：
```bash
which MNNConvert  # 检查是否已安装
```

### 使用方式

#### 方式 1: 批量转换脚本（推荐）

使用批量转换脚本转换目录下所有 ONNX 文件：

```bash
# 转换指定目录下的所有 ONNX 文件
python convert_onnx_to_mnn_batch.py --input_dir motion_file/pm_fall4:v0

# 转换训练日志中的 ONNX 文件
python convert_onnx_to_mnn_batch.py --input_dir logs/rsl_rl/pm1_tracking/2025-12-14_17-37-01
```

#### 方式 2: 使用 onnx_to_mnn.py 脚本

转换单个 ONNX 文件：

```bash
# 从 ONNX 文件转换
python -m mjlab.scripts.onnx_to_mnn \
  --input_file logs/rsl_rl/pm1_tracking/2025-12-14_17-37-01/2025-12-14_17-37-01.onnx \
  --output_file logs/rsl_rl/pm1_tracking/2025-12-14_17-37-01/model.mnn
```

#### 方式 3: 直接使用 MNNConvert 命令行工具

```bash
cd ~/engineai/MNN/build && \
./MNNConvert \
  -f ONNX \
  --modelFile /home/wang22/engineai/mjlab/logs/rsl_rl/pm1_tracking/2025-12-14_17-37-01/2025-12-14_17-37-01.onnx \
  --MNNModel /home/wang22/engineai/mjlab/logs/rsl_rl/pm1_tracking/2025-12-14_17-37-01/model.mnn \
  --bizCode MNN
```

或者如果已安装到系统路径：

```bash
MNNConvert \
  -f ONNX \
  --modelFile model.onnx \
  --MNNModel model.mnn \
  --bizCode MNN
```

### 检查 ONNX 模型信息

```bash
python inspect_onnx.py /home/wang22/engineai/mjlab/logs/rsl_rl/pm1_tracking/2025-12-14_17-37-01/2025-12-14_17-37-01.onnx
```
