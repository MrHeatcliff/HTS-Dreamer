#!/usr/bin/env python3
import csv
import json
import math
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = pathlib.Path("/mnt/disk1/backup_user/dat.tt2/xuance")
REPO = ROOT / "external_baselines/dreamerv3-official"
OUT = REPO / "paper_artifacts/atari_v21_aux_warmup"
DREAMER = ROOT / "logs/external_baselines/dreamerv3_official/full26_size12m"
HTS_V19 = ROOT / "logs/external_baselines/dreamerv3_official_hts_v19_ab/hts_locked_hier_x3"
HTS_V21 = ROOT / "logs/external_baselines/dreamerv3_official_hts_v21_aux_warmup"

TARGET_FRAMES = 440000
BINS = np.linspace(0, TARGET_FRAMES, 21)
COMPLETED_THRESHOLD = BINS[-2]
GAME = "breakout"
STAGE_A_SEEDS = [0, 1, 2]
VARIANTS = [
    ("hts_locked_hier_x3_warmup_50k_raw", "50k_raw", 50000, 12500),
    ("hts_locked_hier_x3_warmup_100k_raw", "100k_raw", 100000, 25000),
]


def read_scores(logdir):
  path = logdir / "scores.jsonl"
  rows = []
  if not path.exists():
    return rows
  for line in path.read_text().splitlines():
    if not line.strip():
      continue
    try:
      item = json.loads(line)
    except Exception:
      continue
    if "step" in item and "episode/score" in item:
      rows.append({
          "raw_frames": float(item["step"]),
          "episode_score": float(item["episode/score"]),
      })
  return rows


def metrics_for_rows(rows):
  if not rows:
    return {
        "latest_raw_frames": 0.0, "episode_score_rows": 0, "completed": False,
        "auc_20bin_mean": math.nan, "final_20pct_mean": math.nan,
        "final_bin_mean": math.nan, "latest_episode_score": math.nan,
        "best_bin_score": math.nan, "bin_means": [math.nan] * 20,
        "bin_counts": [0] * 20,
    }
  xs = np.array([r["raw_frames"] for r in rows], dtype=float)
  ys = np.array([r["episode_score"] for r in rows], dtype=float)
  bin_means, bin_counts = [], []
  for lo, hi in zip(BINS[:-1], BINS[1:]):
    mask = (xs >= lo) & (xs <= hi) if hi == BINS[-1] else (xs >= lo) & (xs < hi)
    vals = ys[mask]
    bin_counts.append(int(vals.size))
    bin_means.append(float(vals.mean()) if vals.size else math.nan)
  arr = np.array(bin_means, dtype=float)
  completed = bool(float(xs.max()) >= COMPLETED_THRESHOLD and not math.isnan(bin_means[-1]))
  return {
      "latest_raw_frames": float(xs.max()),
      "episode_score_rows": int(len(rows)),
      "completed": completed,
      "auc_20bin_mean": float(np.nanmean(arr)) if np.isfinite(arr).any() else math.nan,
      "final_20pct_mean": float(np.nanmean(arr[16:])) if np.isfinite(arr[16:]).any() else math.nan,
      "final_bin_mean": float(arr[-1]) if not math.isnan(arr[-1]) else math.nan,
      "latest_episode_score": float(ys[-1]),
      "best_bin_score": float(np.nanmax(arr)) if np.isfinite(arr).any() else math.nan,
      "bin_means": bin_means,
      "bin_counts": bin_counts,
  }


def add_record(method, condition, seed, source, logdir, extra=None):
  rec = {
      "method": method,
      "condition": condition,
      "game": GAME,
      "seed": seed,
      "source": source,
      "logdir": str(logdir),
      "metric_source_files": str(logdir / "scores.jsonl"),
  }
  rec.update(metrics_for_rows(read_scores(logdir)))
  rec["included_in_completed_primary"] = bool(rec["completed"])
  rec["exclusion_reason"] = "" if rec["completed"] else "missing_or_partial_final_bin"
  if extra:
    rec.update(extra)
  return rec


def dreamer_record(seed):
  candidates = []
  original = DREAMER / GAME / f"seed_{seed}"
  repair = DREAMER / GAME / f"seed_{seed}_repair_no_video"
  if original.exists():
    candidates.append(add_record("DreamerV3", "dreamer_v3_reference", seed, "official_full26_original", original))
  if repair.exists():
    candidates.append(add_record("DreamerV3", "dreamer_v3_reference", seed, "official_full26_repair_no_video", repair))
  if not candidates:
    return add_record("DreamerV3", "dreamer_v3_reference", seed, "missing", original)
  completed = [c for c in candidates if c["completed"]]
  if completed:
    completed.sort(key=lambda c: (c["source"] != "official_full26_original", -c["latest_raw_frames"]))
    return completed[0]
  candidates.sort(key=lambda c: c["latest_raw_frames"], reverse=True)
  return candidates[0]


def v20_hts_record(seed):
  if seed in (0, 2):
    logdir = HTS_V19 / GAME / f"seed_{seed}_retry_no_video"
    source = "V19_retry_no_video"
  else:
    logdir = HTS_V19 / GAME / f"seed_{seed}"
    source = "V19_original"
  return add_record("HTS", "v20_locked_hier_x3", seed, source, logdir)


def v21_record(variant_dir, label, seed, warmup_raw, warmup_actions):
  candidates = [HTS_V21 / variant_dir / GAME / f"seed_{seed}"]
  candidates.extend(sorted((HTS_V21 / variant_dir / GAME).glob(f"seed_{seed}_repair*")))
  records = []
  for logdir in candidates:
    if logdir.exists():
      records.append(add_record("HTS", label, seed, "V21_stage_a", logdir, {
          "warmup_raw_frames": warmup_raw,
          "warmup_agent_actions": warmup_actions,
      }))
  if not records:
    logdir = HTS_V21 / variant_dir / GAME / f"seed_{seed}"
    return add_record("HTS", label, seed, "V21_stage_a_missing", logdir, {
        "warmup_raw_frames": warmup_raw,
        "warmup_agent_actions": warmup_actions,
    })
  completed = [r for r in records if r["completed"]]
  if completed:
    completed.sort(key=lambda r: ("repair" not in pathlib.Path(r["logdir"]).name, -r["latest_raw_frames"]))
    selected = completed[0]
    selected["source"] = "V21_stage_a_repair_selected" if "repair" in pathlib.Path(selected["logdir"]).name else "V21_stage_a"
    return selected
  records.sort(key=lambda r: r["latest_raw_frames"], reverse=True)
  return records[0]


def clean(value):
  if isinstance(value, float) and math.isnan(value):
    return ""
  if isinstance(value, list):
    return json.dumps([clean(x) for x in value])
  return value


def write_json(path, obj):
  path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=clean))


def write_csv(path, rows):
  fields = sorted({k for row in rows for k in row.keys() if k not in ("bin_means", "bin_counts")})
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      writer.writerow({k: clean(row.get(k, "")) for k in fields})


def write_md_table(path, rows):
  if not rows:
    path.write_text("No rows.\n")
    return
  cols = list(rows[0].keys())
  lines = [
      "| " + " | ".join(cols) + " |",
      "| " + " | ".join(["---"] * len(cols)) + " |",
  ]
  for row in rows:
    lines.append("| " + " | ".join(str(clean(row.get(col, ""))) for col in cols) + " |")
  path.write_text("\n".join(lines) + "\n")


def mean(values):
  vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
  return float(statistics.mean(vals)) if vals else math.nan


def aggregate(condition, rows):
  completed = [row for row in rows if row["completed"]]
  return {
      "condition": condition,
      "completed_seeds": ",".join(str(row["seed"]) for row in completed),
      "num_completed": len(completed),
      "auc_20bin_mean": mean([row["auc_20bin_mean"] for row in completed]),
      "final_20pct_mean": mean([row["final_20pct_mean"] for row in completed]),
      "final_bin_mean": mean([row["final_bin_mean"] for row in completed]),
      "latest_episode_score_mean": mean([row["latest_episode_score"] for row in completed]),
  }


def plot_stage_a(all_records):
  centers = (BINS[:-1] + BINS[1:]) / 2
  colors = {
      "dreamer_v3_reference": "#4C78A8",
      "v20_locked_hier_x3": "#F58518",
      "50k_raw": "#54A24B",
      "100k_raw": "#B279A2",
  }
  labels = {
      "dreamer_v3_reference": "DreamerV3",
      "v20_locked_hier_x3": "HTS V20",
      "50k_raw": "HTS warmup 50K raw",
      "100k_raw": "HTS warmup 100K raw",
  }
  fig, ax = plt.subplots(figsize=(6, 5))
  for condition in ["dreamer_v3_reference", "v20_locked_hier_x3", "50k_raw", "100k_raw"]:
    rows = [r for r in all_records if r["condition"] == condition and r["completed"]]
    if not rows:
      continue
    arr = np.array([r["bin_means"] for r in rows], dtype=float)
    m = np.nanmean(arr, axis=0)
    n = np.sum(np.isfinite(arr), axis=0)
    sem = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
    ax.plot(centers, m, lw=2.2, color=colors[condition], label=f"{labels[condition]} (n={len(rows)})")
    ax.fill_between(centers, m - sem, m + sem, color=colors[condition], alpha=0.16, linewidth=0)
  ax.set_title("Breakout Stage A Warmup Screen")
  ax.set_xlabel("Raw frames")
  ax.set_ylabel("Episode score")
  ax.set_xlim(0, TARGET_FRAMES)
  ax.set_xticks([200000, 400000])
  ax.set_xticklabels(["200K", "400K"])
  ax.grid(alpha=0.25)
  ax.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(OUT / "fig_v21_breakout_stage_a_warmup_screen.png", dpi=180)
  fig.savefig(OUT / "fig_v21_breakout_stage_a_warmup_screen.pdf")


def build_stage_b_launcher(selected):
  if not selected:
    return
  variant_dir = "hts_locked_hier_x3_warmup_50k_raw" if selected == "50k_raw" else "hts_locked_hier_x3_warmup_100k_raw"
  config = "hts_warmup_50k_raw" if selected == "50k_raw" else "hts_warmup_100k_raw"
  script = OUT / "launch_v21_stage_b_selected.sh"
  commands = [
      "#!/usr/bin/env bash",
      "set -euo pipefail",
      "mkdir -p /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official/paper_artifacts/atari_v21_aux_warmup/run_logs",
      "echo \"V21 Stage B started at $(date -Is)\"",
  ]
  jobs = [("breakout", 3), ("breakout", 4)] + [("alien", seed) for seed in range(5)]
  for game, seed in jobs:
    run = f"v21__hts_hier_x3_{selected}__{game}__seed{seed}"
    logdir = ROOT / "logs/external_baselines/dreamerv3_official_hts_v21_aux_warmup" / variant_dir / game / f"seed_{seed}"
    commands += [
        f"echo '===== START {run} ====='",
        (
            "cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official && "
            f"export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-5}} WANDB_MODE=online WANDB_PROJECT=hts-wm-atari-dev "
            f"WANDB_GROUP=v21_aux_warmup_breakout WANDB_JOB_TYPE=v21_stage_b_selected "
            f"WANDB_TAGS=v21,aux_warmup,{selected},locked_hier_x3,atari100k,{game},no_video "
            f"WANDB_RUN_NAME={run} XLA_PYTHON_CLIENT_PREALLOCATE=false TMPDIR=/mnt/disk1/backup_user/dat.tt2/xuance/tmp; "
            f"/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python -m dreamerv3.main_hts "
            f"--configs hts_atari100k size12m {config} --task atari100k_{game} --seed {seed} "
            f"--logdir {logdir} --run.steps 110000 --run.envs 1 --run.train_ratio 256 "
            "--run.log_every 250 --run.report_every 999999 --run.log_policy_video False --run.save_every 10000 "
            "--batch_size 16 --batch_length 64 --agent.hts.l_hier 0.3 --agent.report False "
            "--logger.outputs jsonl,scope,wandb --jax.prealloc False --jax.jit True "
            f"2>&1 | tee {OUT}/run_logs/{run}.log"
        ),
        f"echo '===== DONE {run} ====='",
    ]
  commands.append("echo \"V21 Stage B completed at $(date -Is)\"")
  script.write_text("\n".join(commands) + "\n")
  script.chmod(0o755)


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  dreamer = [dreamer_record(seed) for seed in STAGE_A_SEEDS]
  v20 = [v20_hts_record(seed) for seed in STAGE_A_SEEDS]
  v21 = []
  for variant_dir, label, raw, actions in VARIANTS:
    v21.extend(v21_record(variant_dir, label, seed, raw, actions) for seed in STAGE_A_SEEDS)
  all_records = dreamer + v20 + v21

  write_json(OUT / "v21_stage_a_raw_metrics.json", all_records)
  write_csv(OUT / "v21_stage_a_raw_metrics.csv", all_records)
  write_md_table(OUT / "v21_stage_a_raw_metrics.md", all_records)

  aggregates = [
      aggregate("dreamer_v3_reference", dreamer),
      aggregate("v20_locked_hier_x3", v20),
  ]
  for _, label, _, _ in VARIANTS:
    aggregates.append(aggregate(label, [row for row in v21 if row["condition"] == label]))

  ref = {row["condition"]: row for row in aggregates}
  for row in aggregates:
    if row["condition"] in ("50k_raw", "100k_raw"):
      row["delta_auc_vs_v20_hts"] = row["auc_20bin_mean"] - ref["v20_locked_hier_x3"]["auc_20bin_mean"]
      row["delta_final_20pct_vs_v20_hts"] = row["final_20pct_mean"] - ref["v20_locked_hier_x3"]["final_20pct_mean"]
      row["delta_auc_vs_dreamer"] = row["auc_20bin_mean"] - ref["dreamer_v3_reference"]["auc_20bin_mean"]
      row["delta_final_20pct_vs_dreamer"] = row["final_20pct_mean"] - ref["dreamer_v3_reference"]["final_20pct_mean"]
      row["passes_stage_a_rule"] = (
          row["num_completed"] == len(STAGE_A_SEEDS) and
          row["delta_auc_vs_v20_hts"] > 0 and
          row["delta_final_20pct_vs_v20_hts"] > 0 and
          row["delta_auc_vs_dreamer"] >= 0 and
          row["delta_final_20pct_vs_dreamer"] >= 0
      )
    else:
      row["delta_auc_vs_v20_hts"] = ""
      row["delta_final_20pct_vs_v20_hts"] = ""
      row["delta_auc_vs_dreamer"] = ""
      row["delta_final_20pct_vs_dreamer"] = ""
      row["passes_stage_a_rule"] = ""

  write_json(OUT / "v21_stage_a_breakout_screen.json", aggregates)
  write_csv(OUT / "v21_stage_a_breakout_screen.csv", aggregates)
  write_md_table(OUT / "v21_stage_a_breakout_screen.md", aggregates)
  plot_stage_a(all_records)

  candidates = [row for row in aggregates if row["condition"] in ("50k_raw", "100k_raw") and row["passes_stage_a_rule"]]
  selected = ""
  if candidates:
    candidates.sort(key=lambda row: (50000 if row["condition"] == "50k_raw" else 100000))
    selected = candidates[0]["condition"]
  decision = {
      "selected_variant": selected,
      "stage_b_allowed": bool(selected),
      "selection_rule": "Smallest warmup horizon that improves Breakout AUC and final-20 over V20 HTS and is at least competitive with Dreamer on paired seeds 0,1,2.",
      "stage_a_complete": all(row["num_completed"] == len(STAGE_A_SEEDS) for row in aggregates),
      "aggregates": aggregates,
  }
  write_json(OUT / "v21_selected_variant.json", decision)
  (OUT / "v21_selected_variant.md").write_text(
      "# V21 Selected Variant\n\n"
      f"Selected variant: `{selected or 'NONE'}`\n\n"
      f"Stage B allowed: `{bool(selected)}`\n\n"
      "Rule: smallest warmup horizon that improves Breakout AUC and final-20 over V20 HTS "
      "and is at least competitive with DreamerV3 on paired seeds 0,1,2.\n")
  if selected:
    build_stage_b_launcher(selected)
  else:
    (OUT / "v21_no_selection_diagnosis.md").write_text(
        "# V21 No-Selection Diagnosis\n\n"
        "No warmup variant passed Stage A. Do not launch Stage B. Inspect "
        "`v21_stage_a_breakout_screen.md` for AUC/final-20 deltas.\n")
  print(json.dumps(decision, indent=2, sort_keys=True, default=clean))


if __name__ == "__main__":
  main()
