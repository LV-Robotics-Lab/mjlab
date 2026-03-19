#!/usr/bin/env bash
# 顺序执行多个训练任务：前一个跑完后自动跑下一个
# 用法: ./run_train_sequential.sh  或  bash run_train_sequential.sh

set -e  # 任一命令失败则退出（若希望某个失败后继续跑后面的，可注释掉本行）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_CMD="python -m mjlab.scripts.train"
CONFIG="Mjlab-Tracking-Flat-PM1"
# 护具 TSV 文件名（相对 config/pm1/protector_map/），与 train 的 --protector-map-front/back 一致
PROTECTOR_MAP_FRONT="yz_map_front_elbow_knee.tsv"
PROTECTOR_MAP_BACK="yz_map_back_elbow.tsv"
COMMON_ARGS="--env.scene.num-envs 4096 --agent.max_iterations 10000 --protector-map-front ${PROTECTOR_MAP_FRONT} --protector-map-back ${PROTECTOR_MAP_BACK}"

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
  echo ""
  echo "======== [$((i+1))/${#MOTIONS[@]}] 开始: $motion ========"
  $RUN_CMD "$CONFIG" --motion-file "$motion" $COMMON_ARGS
  echo "======== [$((i+1))/${#MOTIONS[@]}] 完成: $motion ========"
done

echo ""
echo "======== 全部 ${#MOTIONS[@]} 个任务已执行完毕 ========"
