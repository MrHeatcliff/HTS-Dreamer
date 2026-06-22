# Breakout Seed0 HTS Levels Sweep

This probe fixes the VICReg covariance scale to the best AUC value from the
previous covariance sweep:

```text
agent.hts.vicreg_cov_scale = 0.001
```

It sweeps:

```text
agent.hts.levels in [2, 4, 6, 8]
```

Each levels preset also changes `topk_per_level`, `strides_coarse_to_fine`,
`beta_hier`, and `alpha_sdyn` to match the number of heads.

Launch:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
CUDA_VISIBLE_DEVICES=5 \
external_baselines/dreamerv3-official/paper_artifacts/atari_levels_sweep/launch_breakout_seed0_levels_sweep.sh
```

Extract:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
.venv/bin/python external_baselines/dreamerv3-official/paper_artifacts/atari_levels_sweep/extract_levels_sweep_metrics.py
```

Outputs:

- `levels_sweep_raw_metrics.{json,csv}`
- `levels_sweep_summary.md`
- `levels_sweep_decision.json`
- `fig_breakout_seed0_levels_sweep_summary.{png,pdf}`
- `fig_breakout_seed0_levels_sweep_curves.{png,pdf}`
