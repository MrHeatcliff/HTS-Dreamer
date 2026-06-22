# V21 Smoke Report

Status: **PASS**

Logdir: `/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v21_aux_warmup/smoke/warmup50k/breakout/seed_0_10k`
W&B: https://wandb.ai/ttdat170703-ho-chi-minh-city-university-of-technology/hts-wm-atari-dev/runs/4in2y0lw

metrics rows: 48
scores rows: 46
warmup train rows: 2
alpha values: `[0.11311999711850632, 0.35551999103107124]`
alpha monotonic: `True`
nonfinite count: `0`

## Required Telemetry

- `train/hts/aux_warmup_alpha`: present in 2/2 warmup rows
- `train/hts/aux_warmup_raw_frames`: present in 2/2 warmup rows
- `train/hts/aux_warmup_agent_actions`: present in 2/2 warmup rows
- `train/hts/aux_warmup_horizon_raw_frames`: present in 2/2 warmup rows
- `train/hts/aux_warmup_horizon_agent_actions`: present in 2/2 warmup rows
- `train/loss/hier_raw`: present in 2/2 warmup rows
- `train/loss/hier_weighted`: present in 2/2 warmup rows
- `train/loss/sdyn_raw`: present in 2/2 warmup rows
- `train/loss/sdyn_weighted`: present in 2/2 warmup rows
- `train/loss/temp_raw`: present in 2/2 warmup rows
- `train/loss/temp_weighted`: present in 2/2 warmup rows
- `train/loss/vc_raw`: present in 2/2 warmup rows
- `train/loss/vc_weighted`: present in 2/2 warmup rows
- `train/loss/sparse_raw`: present in 2/2 warmup rows
- `train/loss/sparse_weighted`: present in 2/2 warmup rows
- `train/hts/coef_hier_effective`: present in 2/2 warmup rows
- `train/hts/coef_sdyn_effective`: present in 2/2 warmup rows
- `train/hts/coef_temp_effective`: present in 2/2 warmup rows
- `train/hts/coef_vc_effective`: present in 2/2 warmup rows
- `train/hts/coef_sparse_effective`: present in 2/2 warmup rows

## Notes
- Smoke 2K only verified W&B/score path; smoke 10K verified HTS warmup train telemetry.
- Policy video was disabled via `--agent.report False --run.log_policy_video False --run.report_every 999999`.
