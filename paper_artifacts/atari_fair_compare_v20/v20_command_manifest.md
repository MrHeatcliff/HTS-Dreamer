# V20 Command Manifest

Selected GPU: `7`

## v20__hts_locked_hier_x3__alien__seed3

Game: `alien` Seed: `3`
Logdir: `/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v20_fair_compare/hts_locked_hier_x3/alien/seed_3`

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official && export CUDA_VISIBLE_DEVICES=7 WANDB_MODE=online WANDB_PROJECT=hts-wm-atari-dev WANDB_GROUP=v20_fair_compare_locked_hier_x3 WANDB_JOB_TYPE=v20_fair_compare WANDB_TAGS=v20,fair_compare,locked_hier_x3,atari100k,alien_breakout,no_video WANDB_RUN_NAME=v20__hts_locked_hier_x3__alien__seed3 XLA_PYTHON_CLIENT_PREALLOCATE=false TMPDIR=/mnt/disk1/backup_user/dat.tt2/xuance/tmp; /mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python -m dreamerv3.main_hts --configs hts_atari100k size12m --task atari100k_alien --seed 3 --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v20_fair_compare/hts_locked_hier_x3/alien/seed_3 --run.steps 110000 --run.envs 1 --run.train_ratio 256 --run.log_every 250 --run.report_every 999999 --run.log_policy_video False --run.save_every 10000 --batch_size 16 --batch_length 64 --agent.hts.l_hier 0.3 --agent.report False --logger.outputs jsonl,scope,wandb --jax.prealloc False --jax.jit True 2>&1 | tee /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official/paper_artifacts/atari_fair_compare_v20/run_logs/v20__hts_locked_hier_x3__alien__seed3.log
```

## v20__hts_locked_hier_x3__alien__seed4

Game: `alien` Seed: `4`
Logdir: `/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v20_fair_compare/hts_locked_hier_x3/alien/seed_4`

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official && export CUDA_VISIBLE_DEVICES=7 WANDB_MODE=online WANDB_PROJECT=hts-wm-atari-dev WANDB_GROUP=v20_fair_compare_locked_hier_x3 WANDB_JOB_TYPE=v20_fair_compare WANDB_TAGS=v20,fair_compare,locked_hier_x3,atari100k,alien_breakout,no_video WANDB_RUN_NAME=v20__hts_locked_hier_x3__alien__seed4 XLA_PYTHON_CLIENT_PREALLOCATE=false TMPDIR=/mnt/disk1/backup_user/dat.tt2/xuance/tmp; /mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python -m dreamerv3.main_hts --configs hts_atari100k size12m --task atari100k_alien --seed 4 --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v20_fair_compare/hts_locked_hier_x3/alien/seed_4 --run.steps 110000 --run.envs 1 --run.train_ratio 256 --run.log_every 250 --run.report_every 999999 --run.log_policy_video False --run.save_every 10000 --batch_size 16 --batch_length 64 --agent.hts.l_hier 0.3 --agent.report False --logger.outputs jsonl,scope,wandb --jax.prealloc False --jax.jit True 2>&1 | tee /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official/paper_artifacts/atari_fair_compare_v20/run_logs/v20__hts_locked_hier_x3__alien__seed4.log
```

## v20__hts_locked_hier_x3__breakout__seed3

Game: `breakout` Seed: `3`
Logdir: `/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v20_fair_compare/hts_locked_hier_x3/breakout/seed_3`

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official && export CUDA_VISIBLE_DEVICES=7 WANDB_MODE=online WANDB_PROJECT=hts-wm-atari-dev WANDB_GROUP=v20_fair_compare_locked_hier_x3 WANDB_JOB_TYPE=v20_fair_compare WANDB_TAGS=v20,fair_compare,locked_hier_x3,atari100k,alien_breakout,no_video WANDB_RUN_NAME=v20__hts_locked_hier_x3__breakout__seed3 XLA_PYTHON_CLIENT_PREALLOCATE=false TMPDIR=/mnt/disk1/backup_user/dat.tt2/xuance/tmp; /mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python -m dreamerv3.main_hts --configs hts_atari100k size12m --task atari100k_breakout --seed 3 --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v20_fair_compare/hts_locked_hier_x3/breakout/seed_3 --run.steps 110000 --run.envs 1 --run.train_ratio 256 --run.log_every 250 --run.report_every 999999 --run.log_policy_video False --run.save_every 10000 --batch_size 16 --batch_length 64 --agent.hts.l_hier 0.3 --agent.report False --logger.outputs jsonl,scope,wandb --jax.prealloc False --jax.jit True 2>&1 | tee /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official/paper_artifacts/atari_fair_compare_v20/run_logs/v20__hts_locked_hier_x3__breakout__seed3.log
```

## v20__hts_locked_hier_x3__breakout__seed4

Game: `breakout` Seed: `4`
Logdir: `/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v20_fair_compare/hts_locked_hier_x3/breakout/seed_4`

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official && export CUDA_VISIBLE_DEVICES=7 WANDB_MODE=online WANDB_PROJECT=hts-wm-atari-dev WANDB_GROUP=v20_fair_compare_locked_hier_x3 WANDB_JOB_TYPE=v20_fair_compare WANDB_TAGS=v20,fair_compare,locked_hier_x3,atari100k,alien_breakout,no_video WANDB_RUN_NAME=v20__hts_locked_hier_x3__breakout__seed4 XLA_PYTHON_CLIENT_PREALLOCATE=false TMPDIR=/mnt/disk1/backup_user/dat.tt2/xuance/tmp; /mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python -m dreamerv3.main_hts --configs hts_atari100k size12m --task atari100k_breakout --seed 4 --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v20_fair_compare/hts_locked_hier_x3/breakout/seed_4 --run.steps 110000 --run.envs 1 --run.train_ratio 256 --run.log_every 250 --run.report_every 999999 --run.log_policy_video False --run.save_every 10000 --batch_size 16 --batch_length 64 --agent.hts.l_hier 0.3 --agent.report False --logger.outputs jsonl,scope,wandb --jax.prealloc False --jax.jit True 2>&1 | tee /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official/paper_artifacts/atari_fair_compare_v20/run_logs/v20__hts_locked_hier_x3__breakout__seed4.log
```
