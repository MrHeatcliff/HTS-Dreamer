# Preflight Atari V20

Status: `BLOCKED_RESOURCE_OR_WANDB_ISSUE`

Selected GPU: `7`

```text
0, NVIDIA GeForce RTX 3090, 23338, 24576, 100
1, NVIDIA GeForce RTX 3090, 23400, 24576, 100
2, NVIDIA GeForce RTX 3090, 15603, 24576, 100
3, NVIDIA GeForce RTX 3090, 3380, 24576, 91
4, NVIDIA GeForce RTX 3090, 14970, 24576, 18
5, NVIDIA GeForce RTX 3090, 14554, 24576, 0
6, NVIDIA GeForce RTX 3090, 14076, 24576, 6
7, NVIDIA GeForce RTX 3090, 2, 24576, 0
```

Disk:

```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1    7.0T  5.7T  952G  86% /mnt/disk1
/dev/sda3       4.4T  3.9T  225G  95% /
```

Memory:

```text
total        used        free      shared  buff/cache   available
Mem:           692Gi       389Gi       6.6Gi       2.5Gi       295Gi       295Gi
Swap:          8.0Gi       5.7Gi       2.3Gi
```

W&B: `{'available': False, 'error': 'ModuleNotFoundError("No module named \'wandb\'")', 'project': 'hts-wm-atari-dev'}`

Existing DreamerV3 processes observed, untouched:

```text
556599    3966 bash external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_audit_20260616/launch_full26_repair_missing_partial.sh
 556602    3966 /mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python -m dreamerv3.main --configs atari100k size12m --task atari100k_breakout --seed 3 --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official/full26_size12m/breakout/seed_3_repair_no_video --run.steps 110000 --run.envs 1 --run.train_ratio 256 --run.log_every 250 --run.report_every 999999 --run.log_policy_video False --run.save_every 10000 --batch_size 16 --batch_length 64 --agent.report False --logger.outputs jsonl,scope,wandb --jax.prealloc False --jax.jit True
 556603    3966 tee /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_audit_20260616/run_logs/DreamerV3-official-size12m-atari100k-breakout-seed3-repair-no-video.log
```

Policy video disabled: `True`. Logdirs use V20 path and do not overwrite V19.
