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

纯mimic向后摔：
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Back_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/h0qf16ob

纯mimic向左摔：1205492990-nus/mjlab/obw2ysrf
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Left_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/obw2ysrf

纯mimic向右摔：1205492990-nus/mjlab/gc5ovv94
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Right_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/gc5ovv94

纯mimic左前摔：1205492990-nus/mjlab/nebt84gj
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/LeftFront_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/nebt84gj

纯mimic左后摔：1205492990-nus/mjlab/x2ohjmc5
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/LeftBack_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/x2ohjmc5

纯mimic右前摔：1205492990-nus/mjlab/mk5mwfqe
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/RightFront_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/mk5mwfqe

纯mimic右后摔：1205492990-nus/mjlab/l7f3pf0x
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/RightBack_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/l7f3pf0x

--wandb-run-path 1205492990-nus/mjlab/6icim82d

cd ~/engineai/MNN/build && \
./MNNConvert \
  -f ONNX \
  --modelFile /home/wang22/engineai/mjlab/logs/rsl_rl/pm1_tracking/2025-12-14_17-37-01/2025-12-14_17-37-01.onnx \
  --MNNModel /home/wang22/engineai/mjlab/logs/rsl_rl/pm1_tracking/2025-12-14_17-37-01/2.mnn \
  --bizCode MNN


python inspect_onnx.py /home/wang22/engineai/mjlab/logs/rsl_rl/pm1_tracking/2025-12-14_17-37-01/2025-12-14_17-37-01.onnx
