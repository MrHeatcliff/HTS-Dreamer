# DreamerV3 Full26 Repair Manifest

Status: `prepared_not_launched`

Audit source: `paper_artifacts/full26_dreamerv3_audit_20260616/full26_audit_report.md`

The original full26 process is stopped. The audit found `118/130` completed seeds and `12` seeds still missing or partial.

## Repair Seeds

| game | seeds | reason |
| --- | --- | --- |
| breakout | 3 | partial at 218760 frames |
| road_runner | 4 | partial at 6252 frames |
| seaquest | 0, 1 | partial at 42480 and 121280 frames |
| seaquest | 2, 3, 4 | not launched |
| up_n_down | 0, 1, 2, 3, 4 | game directory missing, not launched |

## Repair Command

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
CUDA_VISIBLE_DEVICES=0 \
WANDB_PROJECT=HTS-WM-HarmonyDream-Alien-Curve \
external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_audit_20260616/launch_full26_repair_missing_partial.sh
```

The repair launcher writes to `_repair_no_video` logdirs and does not overwrite original partial runs.

Video logging is disabled to avoid W&B GIF/disk failures:

```text
--agent.report False
--run.log_policy_video False
--run.report_every 999999
```
