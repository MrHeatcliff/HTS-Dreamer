#!/usr/bin/env bash
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=online
export WANDB_PROJECT=hts-wm-atari-dev
export WANDB_GROUP=v19_ab_locked_hier_x3
export WANDB_JOB_TYPE=v19_ab_atari_dev_retry_no_video
export WANDB_TAGS=v19_ab,locked_hier_x3,atari100k,size12m,alien_breakout,retry_no_video
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official/paper_artifacts/atari_gate_d2_v19_ab/run_logs
echo '===== START v19_ab__hts_locked_hier_x3__breakout__seed0__retry_no_video ====='
export WANDB_RUN_NAME=v19_ab__hts_locked_hier_x3__breakout__seed0__retry_no_video
/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python -m dreamerv3.main_hts --configs hts_atari100k size12m --task atari100k_breakout --seed 0 --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v19_ab/hts_locked_hier_x3/breakout/seed_0_retry_no_video --run.steps 110000 --run.envs 1 --run.train_ratio 256 --run.log_every 250 --run.report_every 999999 --run.log_policy_video False --run.save_every 10000 --batch_size 16 --batch_length 64 --agent.hts.l_hier 0.3 --agent.report False --logger.outputs jsonl,scope,wandb --jax.prealloc False --jax.jit True 2>&1 | tee /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official/paper_artifacts/atari_gate_d2_v19_ab/run_logs/v19_ab__hts_locked_hier_x3__breakout__seed0__retry_no_video.log
echo '===== DONE v19_ab__hts_locked_hier_x3__breakout__seed0__retry_no_video ====='
echo '===== START v19_ab__hts_locked_hier_x3__breakout__seed1 ====='
export WANDB_RUN_NAME=v19_ab__hts_locked_hier_x3__breakout__seed1
/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python -m dreamerv3.main_hts --configs hts_atari100k size12m --task atari100k_breakout --seed 1 --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v19_ab/hts_locked_hier_x3/breakout/seed_1 --run.steps 110000 --run.envs 1 --run.train_ratio 256 --run.log_every 250 --run.report_every 999999 --run.log_policy_video False --run.save_every 10000 --batch_size 16 --batch_length 64 --agent.hts.l_hier 0.3 --agent.report False --logger.outputs jsonl,scope,wandb --jax.prealloc False --jax.jit True 2>&1 | tee /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official/paper_artifacts/atari_gate_d2_v19_ab/run_logs/v19_ab__hts_locked_hier_x3__breakout__seed1.log
echo '===== DONE v19_ab__hts_locked_hier_x3__breakout__seed1 ====='
echo '===== START v19_ab__hts_locked_hier_x3__breakout__seed2__retry_no_video ====='
export WANDB_RUN_NAME=v19_ab__hts_locked_hier_x3__breakout__seed2__retry_no_video
/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python -m dreamerv3.main_hts --configs hts_atari100k size12m --task atari100k_breakout --seed 2 --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v19_ab/hts_locked_hier_x3/breakout/seed_2_retry_no_video --run.steps 110000 --run.envs 1 --run.train_ratio 256 --run.log_every 250 --run.report_every 999999 --run.log_policy_video False --run.save_every 10000 --batch_size 16 --batch_length 64 --agent.hts.l_hier 0.3 --agent.report False --logger.outputs jsonl,scope,wandb --jax.prealloc False --jax.jit True 2>&1 | tee /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official/paper_artifacts/atari_gate_d2_v19_ab/run_logs/v19_ab__hts_locked_hier_x3__breakout__seed2__retry_no_video.log
echo '===== DONE v19_ab__hts_locked_hier_x3__breakout__seed2__retry_no_video ====='
