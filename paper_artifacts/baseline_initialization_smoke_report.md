# Baseline Initialization Smoke Report

Date: 2026-06-10 Asia/Ho_Chi_Minh

## Official DreamerV3 Anchor

- Status: initialized and previously completed Alien 5-seed campaign in official repo.
- Size preset: `size12m`.
- Observed Alien total params from completed official artifacts: `10,498,772`.
- Artifact root: `/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official/full26_size12m/alien`.

## HTS-WM Official Port

- Status: debug smoke pass.
- Smoke command used CPU debug config and did not launch a paper campaign.
- Smoke logdir: `/tmp/hts_artifact_smoke_a1`.
- Observed debug parameter summary:
  - total: `37,285`
  - HTS module: `25,682`
  - decoder: `6,339`
  - encoder: `3,752`
  - dynamics: `783`
  - actor: `198`
  - reward head: `189`
  - value head: `189`
  - continuation head: `153`
- Note: these are debug-config counts, not size12m paper counts.

## Same-Code P0 Baselines

- `flat_sae`: official-native variant implemented; debug initialization smoke passed at `/tmp/hts_variant_smoke_flat_sae`.
- `flat_mh`: official-native variant implemented; debug initialization smoke passed at `/tmp/hts_variant_smoke_flat_mh`.
- `sgf_style_flat_same_code`: official-native same-code control implemented; debug initialization smoke passed at `/tmp/hts_variant_smoke_sgf_style_flat_same_code`.
- `recon_only_hierarchy`: official-native variant implemented; debug initialization smoke passed at `/tmp/hts_variant_smoke_recon_only_hierarchy`.
- `matryoshka_only`: official-native variant implemented; debug initialization smoke passed at `/tmp/hts_variant_smoke_matryoshka_only`.
- `dense_multistride_no_sparse`: official-native no-sparsity variant implemented; debug initialization smoke passed at `/tmp/hts_variant_smoke_hts_dense_multistride_no_sparse`.
- `larger_flat_param`: analytical width `2648` selected; debug initialization smoke passed at `/tmp/hts_variant_smoke_larger_flat_param`.

The smoke runs verify that modules instantiate, compile, save checkpoints, and enter the training loop under the debug config. They do not replace RT-01..RT-07 regression tests or size12m initialized parameter-count audits.

## Final-Eval Smoke

- Status: pass for artifact plumbing, not a benchmark score.
- Eval logdir: `/tmp/hts_final_eval_smoke_a1`.
- Checkpoint: `/tmp/hts_artifact_smoke_a1/ckpt/20260610T134053F542124`.
- Atari task: `atari100k_pong`.
- Eval episodes: `1`.
- Score: `-21.0`.
- Final eval artifact: `/tmp/hts_final_eval_smoke_a1/paper_artifacts/final_eval.json`.
