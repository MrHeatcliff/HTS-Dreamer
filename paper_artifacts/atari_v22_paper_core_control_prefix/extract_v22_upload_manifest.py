#!/usr/bin/env python3
"""Build upload manifest for V22 Atari deterministic ablations."""

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/mnt/disk1/backup_user/dat.tt2/xuance")
LOG_ROOT = ROOT / "logs/external_baselines/dreamerv3_official_hts_v22_paper_core"
OUT = ROOT / "external_baselines/dreamerv3-official/paper_artifacts/atari_v22_paper_core_control_prefix"
TARGET_FRAMES = 400000
BINS = np.linspace(0, TARGET_FRAMES, 21)
COMPLETED_THRESHOLD = BINS[-2]


RUNS = [
    {
        "condition": "new_joint",
        "display_condition": "new-joint",
        "game": "breakout",
        "seed": 0,
        "logdir": LOG_ROOT / "deterministic_ablation/breakout/joint/seed_0/total_400000raw_tr256_onlineTrue_prefetch1",
    },
    {
        "condition": "new_no_hier",
        "display_condition": "new-noHier",
        "game": "breakout",
        "seed": 0,
        "logdir": LOG_ROOT / "deterministic_ablation/breakout/joint_no_hier_loss/seed_0/total_400000raw_tr256_onlineTrue_prefetch1",
    },
    {
        "condition": "new_2phase50k",
        "display_condition": "new-2phase50k",
        "game": "breakout",
        "seed": 0,
        "logdir": LOG_ROOT / "deterministic_ablation/breakout/two_phase_phase1_50000raw/seed_0/total_400000raw_tr256_onlineTrue_prefetch1",
    },
    {
        "condition": "new_2phase100k",
        "display_condition": "new-2phase100k",
        "game": "breakout",
        "seed": 0,
        "logdir": LOG_ROOT / "deterministic_ablation/breakout/two_phase_phase1_100000raw/seed_0/total_400000raw_tr256_onlineTrue_prefetch1",
    },
    {
        "condition": "new_2phase200k",
        "display_condition": "new-2phase200k",
        "game": "breakout",
        "seed": 0,
        "logdir": LOG_ROOT / "deterministic_ablation/breakout/two_phase_phase1_200000raw/seed_0/total_400000raw_tr256_onlineTrue_prefetch1",
    },
]


def read_scores(logdir):
  rows = []
  path = logdir / "scores.jsonl"
  if not path.exists():
    return rows
  for line in path.read_text().splitlines():
    if not line.strip():
      continue
    try:
      item = json.loads(line)
    except json.JSONDecodeError:
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
        "latest_raw_frames": 0.0,
        "episode_score_rows": 0,
        "completed": False,
        "auc_20bin_mean": math.nan,
        "final_20pct_mean": math.nan,
        "final_bin_mean": math.nan,
        "latest_episode_score": math.nan,
        "best_bin_score": math.nan,
        "bin_means": [math.nan] * 20,
        "bin_counts": [0] * 20,
    }
  xs = np.array([r["raw_frames"] for r in rows], dtype=float)
  ys = np.array([r["episode_score"] for r in rows], dtype=float)
  bin_means = []
  bin_counts = []
  for lo, hi in zip(BINS[:-1], BINS[1:]):
    if hi == BINS[-1]:
      mask = (xs >= lo) & (xs <= hi)
    else:
      mask = (xs >= lo) & (xs < hi)
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
      "best_bin_score": float(np.nanmax(arr)) if np.isfinite(arr).any() else math.nan,
      "bin_means": bin_means,
      "bin_counts": bin_counts,
  }


def clean(v):
  if isinstance(v, float) and math.isnan(v):
    return ""
  if isinstance(v, list):
    return json.dumps([clean(x) for x in v])
  return v


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  records = []
  for spec in RUNS:
    rows = read_scores(spec["logdir"])
    rec = {
        "method": "HTS-new",
        "condition": spec["condition"],
        "display_condition": spec["display_condition"],
        "game": spec["game"],
        "seed": spec["seed"],
        "source": "V22_deterministic_ablation",
        "logdir": str(spec["logdir"]),
        "metric_source_files": str(spec["logdir"] / "scores.jsonl"),
        "included_in_completed_primary": True,
    }
    rec.update(metrics_for_rows(rows))
    rec["exclusion_reason"] = "" if rec["completed"] else "missing_or_partial_final_bin"
    records.append(rec)

  json_path = OUT / "v22_upload_manifest_metrics.json"
  csv_path = OUT / "v22_upload_manifest_metrics.csv"
  md_path = OUT / "v22_upload_manifest_metrics.md"
  json_path.write_text(json.dumps(records, indent=2, sort_keys=True, default=clean))

  fields = sorted({k for r in records for k in r.keys() if k not in ("bin_means", "bin_counts")})
  with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in records:
      writer.writerow({k: clean(r.get(k, "")) for k in fields})

  lines = [
      "# V22 Upload Manifest Metrics",
      "",
      "| condition | game | seed | completed | latest_raw_frames | final_20pct_mean | final_bin_mean | auc_20bin_mean | logdir |",
      "|---|---|---:|---|---:|---:|---:|---:|---|",
  ]
  for r in records:
    lines.append(
        f"| {r['condition']} | {r['game']} | {r['seed']} | {r['completed']} | "
        f"{r['latest_raw_frames']} | {r['final_20pct_mean']} | "
        f"{r['final_bin_mean']} | {r['auc_20bin_mean']} | {r['logdir']} |")
  md_path.write_text("\n".join(lines) + "\n")
  print(f"Wrote {json_path}")
  print(f"Records: {len(records)}")


if __name__ == "__main__":
  main()
