
python -m mjlab.scripts.train Mjlab-Tracking-Flat-PM1 --registry-name e1519767-national-university-of-singapore-org/wandb-registry-motions/pm_fall4 --env.scene.num-envs 4096 --agent.max_iterations 10000
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --wandb-run-path e1519767-national-university-of-singapore/mjlab/bo5t1dmw