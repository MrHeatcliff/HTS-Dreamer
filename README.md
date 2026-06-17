# HTS-WM DreamerV3 Port

This directory is the active codebase for the current HTS-WM experiments. It is a local fork of the official DreamerV3 implementation with additional HTS modules, logging, artifact extraction, and Atari100K comparison scripts.

The parent XuanCe repository is currently used mostly as a workspace and experiment tracker. The DreamerV3 baseline and HTS-WM implementation both run from this directory:

```text
/mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official
```

## Added Files

Core HTS implementation:

```text
dreamerv3/main_hts.py
dreamerv3/hts_agent.py
dreamerv3/hts.py
```

Config and logging changes:

```text
dreamerv3/configs.yaml
dreamerv3/main.py
embodied/run/train.py
embodied/run/eval_only.py
embodied/run/paper_artifacts.py
```

Experiment and analysis utilities:

```text
dreamerv3/atari_gate_d2_v19_ab.py
dreamerv3/extract_full26_reference_v16.py
dreamerv3/synthetic_locked_rerun_v18.py
dreamerv3/synthetic_harness_forensics_v17.py
dreamerv3/synthetic_*.py
paper_artifacts/atari_fair_compare_v20/extract_v20_metrics.py
```

Large generated outputs are ignored by `.gitignore`:

```text
wandb/
paper_artifacts/
logs/
runs/
tmp/
**/ckpt/
**/replay/
**/scope/
*.mp4
*.gif
*.npz
*.pkl
*.ckpt
*.log
```

If a report artifact is small and should be committed, add it explicitly with `git add -f path/to/file`.

## HTS Implementation Summary

HTS-WM augments DreamerV3 by adding hierarchical sparse representation objectives on top of the RSSM representation feature.

Current latent anchor:

```text
anchor name:   rssm_repfeat
anchor source: dreamerv3.rssm.RSSM.loss
anchor dim:    2560 for size12m Atari setup
```

Current locked Atari candidate:

```text
candidate: hts_locked_hier_x3
configs:   hts_atari100k size12m
entrypoint: python -m dreamerv3.main_hts
levels:    6
head_dim:  32
width:     6 x 32 = 192
topk:      8 per level, total 48
strides:   [32, 16, 8, 4, 2, 1]
regime:    joint
l_hier:    0.3
l_sdyn:    0.1
l_temp:    0.01
l_vc:      0.01
l_sparse:  1e-5
```

The HTS loss terms are added to the DreamerV3 training loss while keeping the standard world-model, actor, critic, reward, continuation, and representation losses active.

Logged HTS metrics include:

```text
train/hts/active_ratio
train/hts/mean_abs
train/hts/sparse_l1
train/hts/hier_l1 ... train/hts/hier_l6
train/hts/sdyn_l1 ... train/hts/sdyn_l6
train/hts/sdyn_valid_l1 ... train/hts/sdyn_valid_l6
train/hts/temp_*
train/hts/vicreg_var
train/hts/vicreg_cov
train/hts/total_active_budget
train/hts/total_dictionary_width
train/loss/hts_hier
train/loss/hts_sdyn
train/loss/hts_temp
train/loss/hts_vc
train/loss/hts_sparse
```

## Environment

Use the parent repo's uv environment:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official

/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python - <<'PY'
import jax, wandb
print("jax devices:", jax.devices())
print("wandb:", wandb.__version__)
PY
```

Check GPU availability before launching:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Check W&B before online runs:

```bash
/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python - <<'PY'
import wandb
api = wandb.Api(timeout=20)
print("entity:", api.viewer.entity)
PY
```

## Atari100K Protocol

The current fair-comparison protocol follows the official DreamerV3 Atari100K setup used in this fork:

```text
configs:       atari100k size12m for DreamerV3
configs:       hts_atari100k size12m for HTS-WM
steps:         110000 agent actions
raw frames:    about 440000
action repeat: 4
train_ratio:   256 replayed timesteps per agent action
batch size:    16
batch length:  64
envs:          1
image:         64 x 64 RGB
sticky:        false in atari100k override
reward clip:   false
actions:       needed
lives:         unused
```

In `scores.jsonl`, `step` is raw frame count. Convert to agent actions with:

```text
agent_actions = raw_frames / 4
```

## Run DreamerV3 Baseline

Example for Alien seed 0:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official

CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=online \
WANDB_PROJECT=hts-wm-atari-dev \
WANDB_GROUP=dreamerv3_official_atari100k \
WANDB_RUN_NAME=DreamerV3-official-size12m-atari100k-alien-seed0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python -m dreamerv3.main \
  --configs atari100k size12m \
  --task atari100k_alien \
  --seed 0 \
  --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official/full26_size12m/alien/seed_0 \
  --run.steps 110000 \
  --run.envs 1 \
  --run.train_ratio 256 \
  --run.log_every 250 \
  --run.save_every 10000 \
  --batch_size 16 \
  --batch_length 64 \
  --logger.outputs jsonl,scope,wandb \
  --jax.prealloc False \
  --jax.jit True
```

## Run HTS-WM

Use `dreamerv3.main_hts`.

Example for Alien seed 3 with W&B and policy video disabled:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official

CUDA_VISIBLE_DEVICES=7 \
WANDB_MODE=online \
WANDB_PROJECT=hts-wm-atari-dev \
WANDB_GROUP=v20_fair_compare_locked_hier_x3 \
WANDB_JOB_TYPE=v20_fair_compare \
WANDB_TAGS=v20,fair_compare,locked_hier_x3,atari100k,alien_breakout,no_video \
WANDB_RUN_NAME=v20__hts_locked_hier_x3__alien__seed3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
TMPDIR=/mnt/disk1/backup_user/dat.tt2/xuance/tmp \
/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python -m dreamerv3.main_hts \
  --configs hts_atari100k size12m \
  --task atari100k_alien \
  --seed 3 \
  --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v20_fair_compare/hts_locked_hier_x3/alien/seed_3 \
  --run.steps 110000 \
  --run.envs 1 \
  --run.train_ratio 256 \
  --run.log_every 250 \
  --run.report_every 999999 \
  --run.log_policy_video False \
  --run.save_every 10000 \
  --batch_size 16 \
  --batch_length 64 \
  --agent.hts.l_hier 0.3 \
  --agent.report False \
  --logger.outputs jsonl,scope,wandb \
  --jax.prealloc False \
  --jax.jit True
```

Policy video is disabled because previous long runs hit W&B video/GIF encoding and disk issues. Keep scalar logging online through W&B.

## V20 Fair Comparison

V20 is the current fair Alien/Breakout comparison package:

```text
paper_artifacts/atari_fair_compare_v20/
```

It runs only:

```text
method: hts_locked_hier_x3
games:  alien, breakout
seeds:  3, 4
```

The launcher is:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official

paper_artifacts/atari_fair_compare_v20/launch_v20_hts_missing_seeds.sh
```

Recommended detached launch:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official

tmux new-session -d -s v20_hts_fair_compare \
  "paper_artifacts/atari_fair_compare_v20/launch_v20_hts_missing_seeds.sh > paper_artifacts/atari_fair_compare_v20/v20_launcher.tmux.log 2>&1"
```

Monitor:

```bash
tmux ls
ps -eo pid,ppid,stat,etimes,cmd | grep -E 'launch_v20_hts|dreamerv3.main_hts|v20__hts' | grep -v grep
tail -n 80 paper_artifacts/atari_fair_compare_v20/v20_launcher.tmux.log
```

After runs finish, regenerate metrics and figures:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official

/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python \
  paper_artifacts/atari_fair_compare_v20/extract_v20_metrics.py
```

Main outputs:

```text
paper_artifacts/atari_fair_compare_v20/dreamer_reference_reextract_v20.csv
paper_artifacts/atari_fair_compare_v20/hts_candidate_reextract_v20.csv
paper_artifacts/atari_fair_compare_v20/fair_compare_per_seed_v20.csv
paper_artifacts/atari_fair_compare_v20/fair_compare_aggregate_v20.csv
paper_artifacts/atari_fair_compare_v20/fair_compare_aggregate_v20.md
paper_artifacts/atari_fair_compare_v20/fig_v20_alien_breakout_20bin_curves.png
paper_artifacts/atari_fair_compare_v20/fig_v20_alien_breakout_20bin_curves.pdf
paper_artifacts/atari_fair_compare_v20/fair_comparison_decision_v20.md
paper_artifacts/atari_fair_compare_v20/reference_metric_lineage_audit_v20.md
```

## Metric Protocol

V20 uses a locked metric protocol:

```text
x-axis:             raw frames
target:             440000 raw frames
action_repeat:      4
bin_count:          20
bin_width:          22000 raw frames
primary metrics:    auc_20bin_mean, final_20pct_mean, final_bin_mean
secondary metric:   latest_episode_score
```

Definitions:

```text
auc_20bin_mean:
  Mean of available per-bin episode score means across the 20 fixed bins.

final_20pct_mean:
  Mean score over bins with raw_frames >= 352000.

final_bin_mean:
  Mean score in the final bin [418000, 440000].
```

Do not use `latest_episode_score` alone for headline comparison.

## Log Locations

Official DreamerV3 full-26 logs:

```text
/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official/full26_size12m/
```

HTS V19 logs:

```text
/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v19_ab/hts_locked_hier_x3/
```

HTS V20 logs:

```text
/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v20_fair_compare/hts_locked_hier_x3/
```

Each run commonly contains:

```text
config.yaml
metrics.jsonl
scores.jsonl
paper_artifacts/episode_scores.jsonl
paper_artifacts/latest_train_summary.json
ckpt/
replay/
scope/
```

Use `scores.jsonl` for learning curves and V20 metric extraction.

## Commit Workflow

This directory is its own Git repository. Commit and push it separately from the parent XuanCe repository.

Check remote:

```bash
git remote -v
```

If the remote points to upstream `danijar/dreamerv3`, push to your own fork instead:

```bash
git remote set-url origin https://github.com/MrHeatcliff/dreamerv3-official-hts.git
git push -u origin main
```

Commit code changes without large generated artifacts:

```bash
git add .gitignore README.md \
  dreamerv3/configs.yaml \
  dreamerv3/main.py \
  embodied/run/eval_only.py \
  embodied/run/train.py \
  dreamerv3/*.py \
  embodied/run/paper_artifacts.py

git commit -m "add hts dreamerv3 port"
git push origin main
```

Then update the parent repo pointer:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
git add external_baselines/dreamerv3-official
git commit -m "update dreamerv3 hts subrepo"
git push origin master
```

## Upstream DreamerV3 README

# Mastering Diverse Domains through World Models

A reimplementation of [DreamerV3][paper], a scalable and general reinforcement
learning algorithm that masters a wide range of applications with fixed
hyperparameters.

![DreamerV3 Tasks](https://user-images.githubusercontent.com/2111293/217647148-cbc522e2-61ad-4553-8e14-1ecdc8d9438b.gif)

If you find this code useful, please reference in your paper:

```
@article{hafner2025dreamerv3,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  pages={1--7},
  year={2025},
  publisher={Nature Publishing Group}
}
```

To learn more:

- [Research paper][paper]
- [Project website][website]
- [Twitter summary][tweet]

## DreamerV3

DreamerV3 learns a world model from experiences and uses it to train an actor
critic policy from imagined trajectories. The world model encodes sensory
inputs into categorical representations and predicts future representations and
rewards given actions.

![DreamerV3 Method Diagram](https://user-images.githubusercontent.com/2111293/217355673-4abc0ce5-1a4b-4366-a08d-64754289d659.png)

DreamerV3 masters a wide range of domains with a fixed set of hyperparameters,
outperforming specialized methods. Removing the need for tuning reduces the
amount of expert knowledge and computational resources needed to apply
reinforcement learning.

![DreamerV3 Benchmark Scores](https://github.com/danijar/dreamerv3/assets/2111293/0fe8f1cf-6970-41ea-9efc-e2e2477e7861)

Due to its robustness, DreamerV3 shows favorable scaling properties. Notably,
using larger models consistently increases not only its final performance but
also its data-efficiency. Increasing the number of gradient steps further
increases data efficiency.

![DreamerV3 Scaling Behavior](https://user-images.githubusercontent.com/2111293/217356063-0cf06b17-89f0-4d5f-85a9-b583438c98dd.png)

# Instructions

The code has been tested on Linux and Mac and requires Python 3.11+.

## Docker

You can either use the provided `Dockerfile` that contains instructions or
follow the manual instructions below.

## Manual

Install [JAX][jax] and then the other dependencies:

```sh
pip install -U -r requirements.txt
```

Training script:

```sh
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/{timestamp} \
  --configs crafter \
  --run.train_ratio 32
```

To reproduce results, train on the desired task using the corresponding config,
such as `--configs atari --task atari_pong`.

View results:

```sh
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
```

Scalar metrics are also writting as JSONL files.

# Tips

- All config options are listed in `dreamerv3/configs.yaml` and you can
  override them as flags from the command line.
- The `debug` config block reduces the network size, batch size, duration
  between logs, and so on for fast debugging (but does not learn a good model).
- By default, the code tries to run on GPU. You can switch to CPU or TPU using
  the `--jax.platform cpu` flag.
- You can use multiple config blocks that will override defaults in the
  order they are specified, for example `--configs crafter size50m`.
- By default, metrics are printed to the terminal, appended to a JSON lines
  file, and written as Scope summaries. Other outputs like WandB and
  TensorBoard can be enabled in the training script.
- If you get a `Too many leaves for PyTreeDef` error, it means you're
  reloading a checkpoint that is not compatible with the current config. This
  often happens when reusing an old logdir by accident.
- If you are getting CUDA errors, scroll up because the cause is often just an
  error that happened earlier, such as out of memory or incompatible JAX and
  CUDA versions. Try `--batch_size 1` to rule out an out of memory error.
- Many environments are included, some of which require installing additional
  packages. See the `Dockerfile` for reference.
- To continue stopped training runs, simply run the same command line again and
  make sure that the `--logdir` points to the same directory.

# Disclaimer

This repository contains a reimplementation of DreamerV3 based on the open
source DreamerV2 code base. It is unrelated to Google or DeepMind. The
implementation has been tested to reproduce the official results on a range of
environments.

[jax]: https://github.com/google/jax#pip-installation-gpu-cuda
[paper]: https://arxiv.org/pdf/2301.04104
[website]: https://danijar.com/dreamerv3
[tweet]: https://twitter.com/danijarh/status/1613161946223677441
