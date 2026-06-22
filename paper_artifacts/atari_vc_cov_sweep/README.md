# Breakout Seed0 VICReg Covariance Sweep

This probe sweeps only the covariance coefficient inside the HTS VICReg loss.
The variance margin term is kept at `agent.hts.vicreg_gamma=1.0`, while:

```text
agent.hts.vicreg_cov_scale in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
```

The outer VC loss weight remains the default:

```text
agent.hts.l_vc = 0.01
```

Launch:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
CUDA_VISIBLE_DEVICES=5 \
external_baselines/dreamerv3-official/paper_artifacts/atari_vc_cov_sweep/launch_breakout_seed0_cov_sweep.sh
```

Extract:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
.venv/bin/python external_baselines/dreamerv3-official/paper_artifacts/atari_vc_cov_sweep/extract_cov_sweep_metrics.py
```

Outputs:

- `cov_sweep_raw_metrics.{json,csv}`
- `cov_sweep_summary.md`
- `cov_sweep_decision.json`
- `fig_breakout_seed0_cov_sweep_summary.{png,pdf}`
- `fig_breakout_seed0_cov_sweep_curves.{png,pdf}`
