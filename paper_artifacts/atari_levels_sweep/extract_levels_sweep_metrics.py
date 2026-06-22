#!/usr/bin/env python3
import csv
import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = pathlib.Path("/mnt/disk1/backup_user/dat.tt2/xuance")
REPO = ROOT / "external_baselines/dreamerv3-official"
OUT = REPO / "paper_artifacts/atari_levels_sweep"
LOG_ROOT = ROOT / "logs/external_baselines/dreamerv3_official_hts_levels_sweep/breakout_seed0_cov0p001"
LEVELS = [2, 4, 6, 8]
TARGET_FRAMES = 440000
BINS = np.linspace(0, TARGET_FRAMES, 21)
COMPLETED_THRESHOLD = BINS[-2]


def read_jsonl(path):
  if not path.exists():
    return []
  rows = []
  for line in path.read_text().splitlines():
    if not line.strip():
      continue
    try:
      rows.append(json.loads(line))
    except Exception:
      pass
  return rows


def read_scores(logdir):
  rows = []
  for item in read_jsonl(logdir / "scores.jsonl"):
    if "step" in item and "episode/score" in item:
      rows.append({
          "raw_frames": float(item["step"]),
          "episode_score": float(item["episode/score"]),
      })
  if rows:
    return rows
  for item in read_jsonl(logdir / "paper_artifacts/episode_scores.jsonl"):
    frame = item.get("frames", item.get("realized_frames", item.get("step")))
    score = item.get("episode_score")
    if frame is not None and score is not None:
      rows.append({"raw_frames": float(frame), "episode_score": float(score)})
  return rows


def read_last_metrics(logdir):
  rows = read_jsonl(logdir / "metrics.jsonl")
  return rows[-1] if rows else {}


def read_config_fields(logdir):
  path = logdir / "config.yaml"
  text = path.read_text() if path.exists() else ""
  fields = {}
  for key in ["levels", "head_dim", "topk_per_level", "strides_coarse_to_fine", "vicreg_cov_scale"]:
    marker = f"{key}:"
    for line in text.splitlines():
      if marker in line:
        fields[f"config_{key}"] = line.split(marker, 1)[1].strip()
        break
  return fields


def metrics_for_rows(rows):
  if not rows:
    return {
        "latest_raw_frames": 0.0, "episode_score_rows": 0, "completed": False,
        "auc_20bin_mean": math.nan, "final_20pct_mean": math.nan,
        "final_bin_mean": math.nan, "latest_episode_score": math.nan,
        "bin_means": [math.nan] * 20, "bin_counts": [0] * 20,
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
  return {
      "latest_raw_frames": float(xs.max()),
      "episode_score_rows": int(len(rows)),
      "completed": bool(float(xs.max()) >= COMPLETED_THRESHOLD and not math.isnan(bin_means[-1])),
      "auc_20bin_mean": float(np.nanmean(arr)) if np.isfinite(arr).any() else math.nan,
      "final_20pct_mean": float(np.nanmean(arr[16:])) if np.isfinite(arr[16:]).any() else math.nan,
      "final_bin_mean": float(arr[-1]) if not math.isnan(arr[-1]) else math.nan,
      "latest_episode_score": float(ys[-1]),
      "bin_means": bin_means,
      "bin_counts": bin_counts,
  }


def clean(value):
  if isinstance(value, float) and math.isnan(value):
    return ""
  if isinstance(value, list):
    return json.dumps([clean(x) for x in value])
  return value


def write_csv(path, rows):
  fields = sorted({k for r in rows for k in r.keys() if k not in ("bin_means", "bin_counts")})
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      writer.writerow({k: clean(row.get(k, "")) for k in fields})


def write_md(path, rows):
  cols = [
      "levels", "completed", "latest_raw_frames", "auc_20bin_mean",
      "final_20pct_mean", "final_bin_mean", "latest_episode_score",
      "last_total_dictionary_width", "last_total_active_budget",
      "last_vicreg_cov_scale", "logdir",
  ]
  lines = [
      "| " + " | ".join(cols) + " |",
      "| " + " | ".join(["---"] * len(cols)) + " |",
  ]
  for row in rows:
    lines.append("| " + " | ".join(str(clean(row.get(c, ""))) for c in cols) + " |")
  path.write_text("\n".join(lines) + "\n")


def plot(rows):
  done = [r for r in rows if r["completed"]]
  if not done:
    return
  levels = np.array([r["levels"] for r in done], dtype=float)
  auc = np.array([r["auc_20bin_mean"] for r in done], dtype=float)
  final20 = np.array([r["final_20pct_mean"] for r in done], dtype=float)
  finalbin = np.array([r["final_bin_mean"] for r in done], dtype=float)
  fig, ax = plt.subplots(figsize=(6, 4.5))
  ax.plot(levels, auc, marker="o", label="AUC 20-bin")
  ax.plot(levels, final20, marker="o", label="Final 20%")
  ax.plot(levels, finalbin, marker="o", label="Final bin")
  ax.set_xlabel("HTS levels")
  ax.set_ylabel("Breakout episode score")
  ax.set_xticks(LEVELS)
  ax.grid(alpha=0.25)
  ax.legend()
  fig.tight_layout()
  fig.savefig(OUT / "fig_breakout_seed0_levels_sweep_summary.png", dpi=180)
  fig.savefig(OUT / "fig_breakout_seed0_levels_sweep_summary.pdf")

  centers = (BINS[:-1] + BINS[1:]) / 2
  fig, ax = plt.subplots(figsize=(7, 4.5))
  cmap = plt.get_cmap("viridis")
  for i, row in enumerate(done):
    ax.plot(
        centers, np.array(row["bin_means"], dtype=float),
        color=cmap(i / max(len(done) - 1, 1)), lw=2,
        label=f"L={row['levels']}")
  ax.set_xlabel("Raw frames")
  ax.set_ylabel("Episode score")
  ax.set_xlim(0, TARGET_FRAMES)
  ax.set_xticks([200000, 400000])
  ax.set_xticklabels(["200K", "400K"])
  ax.grid(alpha=0.25)
  ax.legend(fontsize=9)
  fig.tight_layout()
  fig.savefig(OUT / "fig_breakout_seed0_levels_sweep_curves.png", dpi=180)
  fig.savefig(OUT / "fig_breakout_seed0_levels_sweep_curves.pdf")


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  rows = []
  for levels in LEVELS:
    logdir = LOG_ROOT / f"levels_{levels}" / "seed_0"
    record = {
        "game": "breakout",
        "seed": 0,
        "condition": "hts_levels_sweep_cov0p001",
        "levels": levels,
        "logdir": str(logdir),
    }
    record.update(read_config_fields(logdir))
    record.update(metrics_for_rows(read_scores(logdir)))
    last = read_last_metrics(logdir)
    record.update({
        "last_total_dictionary_width": last.get("train/hts/total_dictionary_width", ""),
        "last_total_active_budget": last.get("train/hts/total_active_budget", ""),
        "last_vicreg_cov_scale": last.get("train/hts/vicreg_cov_scale", ""),
        "last_vicreg_var": last.get("train/hts/vicreg_var", ""),
        "last_vicreg_cov": last.get("train/hts/vicreg_cov", ""),
    })
    rows.append(record)
  (OUT / "levels_sweep_raw_metrics.json").write_text(json.dumps(rows, indent=2, sort_keys=True, default=clean))
  write_csv(OUT / "levels_sweep_raw_metrics.csv", rows)
  write_md(OUT / "levels_sweep_summary.md", rows)
  plot(rows)
  decision = {
      "all_complete": all(r["completed"] for r in rows),
      "best_auc": max((r for r in rows if r["completed"]), key=lambda r: r["auc_20bin_mean"], default={}),
      "best_final_20pct": max((r for r in rows if r["completed"]), key=lambda r: r["final_20pct_mean"], default={}),
      "best_final_bin": max((r for r in rows if r["completed"]), key=lambda r: r["final_bin_mean"], default={}),
  }
  (OUT / "levels_sweep_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True, default=clean))
  print(json.dumps(decision, indent=2, sort_keys=True, default=clean))


if __name__ == "__main__":
  main()
