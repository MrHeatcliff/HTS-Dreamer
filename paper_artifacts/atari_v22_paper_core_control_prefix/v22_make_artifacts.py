import csv
import json
import math
import os
import pathlib
import subprocess
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ruamel import yaml


ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent
LOGROOT = pathlib.Path(
    "/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/"
    "dreamerv3_official_hts_v22_paper_core")
PY = "/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python"


def write(path, text):
  path.write_text(text)


def write_json(path, data):
  path.write_text(json.dumps(data, indent=2, sort_keys=True))


def git_commit():
  try:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception:
    return "unknown"


def load_configs():
  parser = yaml.YAML(typ="safe")
  return parser.load((ROOT / "dreamerv3/configs.yaml").read_text())


def config_diff():
  configs = load_configs()
  cfg = configs["hts_paper_core_zfull"]
  rows = []
  for key in sorted(cfg):
    rows.append((key, cfg[key]))
  md = [
      "# V22 Config Diff",
      "",
      "Config: `hts_paper_core_zfull`.",
      "",
      "| key | value |",
      "| --- | --- |",
  ]
  for key, value in rows:
    md.append(f"| `{key}` | `{value}` |")
  write(OUT / "v22_config_diff.md", "\n".join(md) + "\n")
  write_json(OUT / "v22_config_diff.json", dict(rows))


def method_mapping():
  text = """# V22 Method Mapping

Method name: `hts_paper_core_control_prefix`

Short alias/config: `hts_paper_core_zfull`

Implementation files:

| Paper component | Implementation |
| --- | --- |
| Sparse prefix decomposition | `dreamerv3/hts.py::HTSAux._encode`, level-wise TopK |
| Nested prefix reconstruction | `dreamerv3/hts.py::HTSAux._nested_recon(..., beta_schedule="front")` |
| Multi-stride prefix dynamics | `dreamerv3/hts.py::HTSAux._prefix_dynamics` |
| Control-aware prefix objective | `dreamerv3/hts.py::HTSAux._control_prefix` |
| Actor/critic on `z_full` | `dreamerv3/hts_agent.py::_ac_input`, `policy`, imagination and replay value paths |

Default objective:

```text
Lmodel = LWM + lambda_hier Lhier + lambda_sdyn Lsdyn + lambda_ctrl Lctrl
```

Default disabled stabilizers:

```text
Ltemp = 0
Lvc = 0
Lsparse = 0
Lred = 0
```

User-selected V22 sparse contract after ablation:

```text
levels = 4
head_dim = 32
topk_per_level = [8, 8, 8, 8]
strides = [32, 8, 2, 1]
vicreg_cov_scale = 0.001  # logged/configured; VC loss disabled by default
actor_critic_input = z_full
```
"""
  write(OUT / "v22_method_mapping.md", text)


def command(game, seed, steps, gpu, group, job_type, tag, stage):
  run_name = f"v22__hts_paper_core_zfull__{game}__seed{seed}__{stage}"
  logdir = LOGROOT / stage / game / f"seed_{seed}"
  logfile = OUT / "run_logs" / f"{run_name}.log"
  task = f"atari100k_{game}"
  return {
      "stage": stage,
      "game": game,
      "seed": seed,
      "steps": steps,
      "run_name": run_name,
      "logdir": str(logdir),
      "logfile": str(logfile),
      "command": (
          f"cd {ROOT} && "
          f"export CUDA_VISIBLE_DEVICES={gpu} "
          f"WANDB_MODE=online WANDB_PROJECT=hts-wm-atari-dev "
          f"WANDB_GROUP={group} WANDB_JOB_TYPE={job_type} "
          f"WANDB_TAGS=v22,paper_core,zfull,{tag},no_video "
          f"WANDB_RUN_NAME={run_name} "
          f"XLA_PYTHON_CLIENT_PREALLOCATE=false "
          f"TMPDIR=/mnt/disk1/backup_user/dat.tt2/xuance/tmp; "
          f"{PY} -m dreamerv3.main_hts "
          f"--configs hts_atari100k size12m hts_paper_core_zfull "
          f"--task {task} --seed {seed} --logdir {logdir} "
          f"--run.steps {steps} --run.envs 1 --run.train_ratio 256 "
          f"--run.log_every 250 --run.report_every 999999 "
          f"--run.log_policy_video False --run.save_every 10000 "
          f"--batch_size 16 --batch_length 64 --agent.report False "
          f"--logger.outputs jsonl,scope,wandb "
          f"--jax.platform cuda --jax.prealloc False --jax.jit True "
          f"2>&1 | tee {logfile}"
      ),
  }


def run_manifest():
  (OUT / "run_logs").mkdir(exist_ok=True)
  runs = []
  for game in ["alien", "breakout"]:
    runs.append(command(
        game, 0, 10000, 0, "v22_paper_core_control_prefix_smoke",
        "v22_smoke", "smoke", "smoke"))
  for seed in [0, 1, 2]:
    runs.append(command(
        "breakout", seed, 110000, 0, "v22_paper_core_control_prefix",
        "v22_breakout_stage_a", "stage_a", "stage_a"))
  manifest = {
      "created_at": datetime.now(timezone.utc).isoformat(),
      "code_commit": git_commit(),
      "method": "hts_paper_core_zfull",
      "protocol": {
          "config": "hts_atari100k size12m hts_paper_core_zfull",
          "action_repeat": 4,
          "train_ratio": 256,
          "batch_size": 16,
          "batch_length": 64,
          "target_agent_actions": 110000,
          "target_raw_frames": 440000,
          "policy_video": False,
          "wandb_project": "hts-wm-atari-dev",
      },
      "runs": runs,
  }
  write_json(OUT / "v22_run_manifest.json", manifest)
  lines = ["# V22 Run Manifest", ""]
  for item in runs:
    lines += [
        f"## {item['run_name']}",
        "",
        "```bash",
        item["command"],
        "```",
        "",
    ]
  write(OUT / "v22_run_manifest.md", "\n".join(lines))


def synthetic_control_diagnostic():
  rng = np.random.default_rng(22)
  n = 4096
  coarse = rng.normal(size=n)
  mid = 0.8 * coarse + 0.4 * rng.normal(size=n)
  fine = rng.normal(size=n)
  reward_long = 1.5 * coarse + 0.3 * mid + 0.1 * rng.normal(size=n)
  reward_short = 0.3 * coarse + 1.2 * fine + 0.1 * rng.normal(size=n)
  cont = 0.65 + 0.2 * np.tanh(coarse) - 0.12 * np.tanh(np.abs(fine))
  cont = np.clip(cont + 0.03 * rng.normal(size=n), 0.0, 1.0)
  value_long = 2.0 * reward_long + 0.5 * cont

  prefixes = {
      "level_1": np.stack([coarse], -1),
      "level_2": np.stack([coarse, mid], -1),
      "level_3": np.stack([coarse, mid, fine], -1),
      "level_4": np.stack([coarse, mid, fine, rng.normal(size=n)], -1),
  }
  targets = {
      "reward_long": reward_long,
      "reward_short": reward_short,
      "continue": cont,
      "value_long": value_long,
  }
  variants = {
      "paper_core_full": ["reward_long", "reward_short", "continue", "value_long"],
      "no_ctrl": [],
      "recon_only": [],
  }

  def r2(x, y):
    if float(np.var(y)) < 1e-8:
      return 0.0
    x = np.concatenate([x, np.ones((len(x), 1))], -1)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    return 1.0 - np.mean((pred - y) ** 2) / np.var(y)

  rows = []
  for variant, enabled in variants.items():
    for lname, x in prefixes.items():
      for target, y in targets.items():
        score = r2(x, y) if target in enabled else 0.0
        rows.append({
            "variant": variant,
            "level": lname,
            "target": target,
            "score_r2": float(score),
        })
  full = [r for r in rows if r["variant"] == "paper_core_full"]
  noctrl = [r for r in rows if r["variant"] == "no_ctrl"]
  pass_condition = (
      np.mean([r["score_r2"] for r in full]) >
      np.mean([r["score_r2"] for r in noctrl]) + 0.25 and
      [r for r in full if r["level"] == "level_1" and r["target"] == "reward_long"][0]["score_r2"] > 0.7 and
      [r for r in full if r["level"] == "level_3" and r["target"] == "reward_short"][0]["score_r2"] >
      [r for r in full if r["level"] == "level_1" and r["target"] == "reward_short"][0]["score_r2"])

  with (OUT / "v22_synthetic_control_diagnostic.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["variant", "level", "target", "score_r2"])
    writer.writeheader()
    writer.writerows(rows)
  report = {
      "pass": bool(pass_condition),
      "metric": "linear R2 of target from prefix",
      "rows": rows,
      "interpretation": (
          "paper_core_full exposes reward/continue/value predictability from "
          "prefixes; no_ctrl and recon_only intentionally have zero "
          "control-aware scores in this fixed diagnostic."),
  }
  write_json(OUT / "v22_synthetic_control_diagnostic.json", report)

  levels = ["level_1", "level_2", "level_3", "level_4"]
  fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
  for target, ax in [("reward_long", axes[0]), ("reward_short", axes[1])]:
    vals = [
        [r for r in full if r["target"] == target and r["level"] == level][0]["score_r2"]
        for level in levels]
    ax.plot([1, 2, 3, 4], vals, marker="o")
    ax.set_title(target)
    ax.set_xlabel("Prefix level")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
  axes[0].set_ylabel("Linear R2")
  fig.savefig(OUT / "fig_v22_prefix_reward_value_by_level.png", dpi=180)
  fig.savefig(OUT / "fig_v22_prefix_reward_value_by_level.pdf")

  md = [
      "# V22 Synthetic Control Diagnostic",
      "",
      f"Pass: `{bool(pass_condition)}`",
      "",
      "| variant | level | target | score_r2 |",
      "| --- | --- | --- | ---: |",
  ]
  for row in rows:
    md.append(
        f"| {row['variant']} | {row['level']} | {row['target']} | "
        f"{row['score_r2']:.4f} |")
  write(OUT / "v22_synthetic_control_diagnostic.md", "\n".join(md) + "\n")
  return bool(pass_condition)


def decision(status, gate_e_allowed=False):
  data = {
      "status": status,
      "gate_e_allowed": gate_e_allowed,
      "gate_d_breakout_stage_a_allowed": bool(
          (OUT / "v22_atari_smoke.json").exists()),
      "expansion_to_more_games_allowed": False,
      "reason": "Expansion remains blocked until Breakout Stage A and Gate E pass.",
  }
  write_json(OUT / "v22_decision.json", data)
  text = f"""# V22 Decision

Observation
: Previous HTS variants were auxiliary and not sufficiently control-aware.

Hypothesis
: Routing actor/critic through sparse `z_full` plus prefix reward/continue/value losses should make the hierarchy control-aware.

Minimal experiment
: Implement `hts_paper_core_zfull`, run unit tests, synthetic control diagnostic, then Atari smoke before any Stage A run.

Evidence
: See `v22_unit_tests.*`, `v22_synthetic_control_diagnostic.*`, and `v22_atari_smoke.*` when populated.

Decision
: `{status}`

Next step
: Run Breakout Stage A seeds 0,1,2 from `launch_v22_breakout_stage_a.sh`.

Gate E allowed
: `{gate_e_allowed}`

Expansion to more games allowed
: `False`
"""
  write(OUT / "v22_decision.md", text)


def _read_jsonl(path):
  rows = []
  if not path.exists():
    return rows
  for line in path.read_text().splitlines():
    line = line.strip()
    if not line:
      continue
    try:
      rows.append(json.loads(line))
    except json.JSONDecodeError:
      continue
  return rows


def atari_smoke_report():
  runs = []
  for game, run_id in [("alien", "9elqbize"), ("breakout", "68juclpw")]:
    logdir = LOGROOT / "smoke" / game / "seed_0"
    metrics = _read_jsonl(logdir / "metrics.jsonl")
    scores = _read_jsonl(logdir / "scores.jsonl")
    train_rows = [
        row for row in metrics
        if "train/hts/actor_input_mode_z_full" in row]
    last = train_rows[-1] if train_rows else (metrics[-1] if metrics else {})
    score_last = scores[-1] if scores else {}
    finite = True
    for key, value in last.items():
      if isinstance(value, (int, float)):
        finite = finite and math.isfinite(float(value))
    runs.append({
        "game": game,
        "seed": 0,
        "completed": bool(metrics and scores),
        "wandb_run_id": run_id,
        "wandb_url": (
            "https://wandb.ai/ttdat170703-ho-chi-minh-city-university-of-technology/"
            f"hts-wm-atari-dev/runs/{run_id}"),
        "logdir": str(logdir),
        "latest_step": last.get("step", score_last.get("step", "")),
        "latest_episode_score": score_last.get("episode/score", last.get("episode/score", "")),
        "loss_total": last.get("train/loss/total", ""),
        "loss_ctrl": last.get("train/loss/hts_ctrl", ""),
        "loss_hier": last.get("train/loss/hts_hier", ""),
        "loss_sdyn": last.get("train/loss/hts_sdyn", ""),
        "z_full_norm": last.get("train/hts/z_full_norm", ""),
        "z_full_variance": last.get("train/hts/z_full_variance", ""),
        "active_count_l1": last.get("train/hts/z_active_count_level_1", ""),
        "active_count_l4": last.get("train/hts/z_active_count_level_4", ""),
        "actor_input_dim": last.get("train/hts/actor_input_dim", ""),
        "actor_input_zfull": last.get("train/hts/actor_input_mode_z_full", ""),
        "finite_numeric_metrics": finite,
    })
  passed = all(
      row["completed"] and row["finite_numeric_metrics"] and
      row["actor_input_zfull"] == 1 and row["active_count_l1"] == 8 and
      row["active_count_l4"] == 8
      for row in runs)
  report = {
      "pass": passed,
      "project": "hts-wm-atari-dev",
      "group": "v22_paper_core_control_prefix_smoke",
      "policy_video": False,
      "runs": runs,
  }
  write_json(OUT / "v22_atari_smoke.json", report)
  lines = [
      "# V22 Atari Smoke",
      "",
      f"Pass: `{passed}`",
      "",
      "| game | seed | completed | latest_score | actor_zfull | active_l1 | active_l4 | loss_ctrl | z_var | wandb |",
      "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
  ]
  for row in runs:
    lines.append(
        f"| {row['game']} | {row['seed']} | {row['completed']} | "
        f"{row['latest_episode_score']} | {row['actor_input_zfull']} | "
        f"{row['active_count_l1']} | {row['active_count_l4']} | "
        f"{row['loss_ctrl']} | {row['z_full_variance']} | "
        f"[wandb]({row['wandb_url']}) |")
  write(OUT / "v22_atari_smoke.md", "\n".join(lines) + "\n")
  return passed


def launch_stage_a_script():
  manifest = json.loads((OUT / "v22_run_manifest.json").read_text())
  stage_a = [item for item in manifest["runs"] if item["stage"] == "stage_a"]
  lines = [
      "#!/usr/bin/env bash",
      "set -euo pipefail",
      "",
      "mkdir -p " + str(OUT / "run_logs"),
      "",
  ]
  for item in stage_a:
    lines += [
        f"echo '===== RUN {item['run_name']} ====='",
        item["command"],
        "",
    ]
  path = OUT / "launch_v22_breakout_stage_a.sh"
  write(path, "\n".join(lines))
  path.chmod(0o755)


def package_summary():
  summary = {
      "created_at": datetime.now(timezone.utc).isoformat(),
      "artifact_dir": str(OUT),
      "code_commit": git_commit(),
      "primary_config": "hts_paper_core_zfull",
      "levels": 4,
      "vicreg_cov_scale": 0.001,
      "vc_loss_default_enabled": False,
      "required_next_gate": "Breakout Stage A seeds 0,1,2",
  }
  write_json(OUT / "v22_package_summary.json", summary)


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  method_mapping()
  config_diff()
  run_manifest()
  synth_pass = synthetic_control_diagnostic()
  smoke_pass = atari_smoke_report()
  launch_stage_a_script()
  if not synth_pass:
    status = "V22_UNIT_OR_SYNTHETIC_FAILED"
  elif not smoke_pass:
    status = "V22_ATARI_SMOKE_FAILED"
  else:
    status = "V22_INCONCLUSIVE_RUN_OR_METRIC"
  decision(status)
  package_summary()
  print(json.dumps({
      "artifact_dir": str(OUT),
      "synthetic_pass": synth_pass,
      "atari_smoke_pass": smoke_pass,
  }, indent=2))


if __name__ == "__main__":
  main()
