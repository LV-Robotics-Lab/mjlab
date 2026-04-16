#!/usr/bin/env bash
# 顺序执行 8 个方向 fine-tune：前一个跑完后自动跑下一个
# 用法: 直接编辑本文件中的 RUN_PATHS 后执行
#   bash run_train_sequential_finetune.sh

set -e  # 任一命令失败则退出（若希望某个失败后继续跑后面的，可注释掉本行）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_CMD="python -m mjlab.scripts.train"
CONFIG="Mjlab-Tracking-Flat-PM1"
# 护具 TSV 文件名（相对 config/pm1/protector_map/），与 train 的 --protector-map-front/back 一致
# PROTECTOR_MAP_FRONT="yz_map_front_elbow_knee.tsv"
# PROTECTOR_MAP_BACK="yz_map_back_elbow.tsv"
# COMMON_ARGS="--env.scene.num-envs 4096 --agent.max_iterations 10000 --protector-map-front ${PROTECTOR_MAP_FRONT} --protector-map-back ${PROTECTOR_MAP_BACK}"

# 不用护具 map 时：注释掉上面 PROTECTOR_MAP_* 与 COMMON_ARGS 一行，改用下面这行（reduce_contact_force 用原始接触力）：
COMMON_ARGS="--env.scene.num-envs 4096 --agent.max_iterations 10000 --use-protector-map False"
# 仍走 map 逻辑但厚度为 0 时，可改用：yz_map_front_zero.tsv / yz_map_back_zero.tsv 作为 PROTECTOR_MAP_*。

# 8 个方向对应的 wandb run path（必须与 MOTIONS 顺序一致）
RUN_PATHS=(
  "e1519767-national-university-of-singapore/mjlab/rnxjg8iy"  # front
  "e1519767-national-university-of-singapore/mjlab/vl2m5u9m"  # back
  "e1519767-national-university-of-singapore/mjlab/7v56ducf"  # left
  "e1519767-national-university-of-singapore/mjlab/2v8i1okj"  # right
  "e1519767-national-university-of-singapore/mjlab/5bdtdg57"  # left_front
  "e1519767-national-university-of-singapore/mjlab/wy2mg23l"  # left_back
  "e1519767-national-university-of-singapore/mjlab/ryjlz229"  # right_front
  "e1519767-national-university-of-singapore/mjlab/srz6filb"  # right_back
)

# 要顺序执行的 motion 文件列表（可自行增删改）
MOTIONS=(
  "motion_file/pm_fall4:v0/toFront_1_converted_50fps.npz"
  "motion_file/pm_fall4:v0/toBack_1_converted_50fps.npz"
  "motion_file/pm_fall4:v0/toLeft_1_converted_50fps.npz"
  "motion_file/pm_fall4:v0/toRight_1_converted_50fps.npz"
  "motion_file/pm_fall4:v0/toLeftFront_1_converted_50fps.npz"
  "motion_file/pm_fall4:v0/toLeftBack_1_converted_50fps.npz"
  "motion_file/pm_fall4:v0/toRightFront_1_converted_50fps.npz"
  "motion_file/pm_fall4:v0/toRightBack_1_converted_50fps.npz"
)

echo "======== 共 ${#MOTIONS[@]} 个任务，按顺序执行 ========"

for i in "${!MOTIONS[@]}"; do
  motion="${MOTIONS[$i]}"
  run_path="${RUN_PATHS[$i]}"
  echo ""
  echo "======== [$((i+1))/${#MOTIONS[@]}] 开始: $motion ========"
  echo "======== 使用 run_path: $run_path ========"
  $RUN_CMD "$CONFIG" --motion-file "$motion" --agent.resume True --wandb-run-path "$run_path" $COMMON_ARGS
  echo "======== [$((i+1))/${#MOTIONS[@]}] 完成: $motion ========"
done

echo ""
echo "======== 全部 ${#MOTIONS[@]} 个任务已执行完毕 ========"
