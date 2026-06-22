# HTS No-VC Alien/Breakout Probe

This probe runs HTS `locked_hier_x3` with the VICReg variance-covariance loss disabled:

```text
--configs hts_atari100k size12m hts_no_vc
agent.hts.l_vc = 0.0
```

Scope:

- Games: `alien`, `breakout`
- Seeds: `0,1,2,3,4`
- W&B project: `hts-wm-atari-dev`
- W&B group: `no_vc_alien_breakout_probe`
- Videos disabled via `--agent.report False --run.log_policy_video False`
- Metric protocol: 20 bins over raw frames `0..440000`, final-20%, final-bin.

Launch:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
CUDA_VISIBLE_DEVICES=5 \
external_baselines/dreamerv3-official/paper_artifacts/atari_no_vc_ablation/launch_no_vc_alien_breakout_5seed.sh
```

Extract:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
.venv/bin/python external_baselines/dreamerv3-official/paper_artifacts/atari_no_vc_ablation/extract_no_vc_metrics.py
```

Outputs:

- `no_vc_raw_metrics.{json,csv,md}`
- `no_vc_aggregate.{json,csv,md}`
- `no_vc_decision.json`
- `fig_no_vc_alien_breakout_20bin_curves.{png,pdf}`
