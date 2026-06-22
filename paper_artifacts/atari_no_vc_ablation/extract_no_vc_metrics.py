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
OUT = REPO / "paper_artifacts/atari_no_vc_ablation"
DREAMER = ROOT / "logs/external_baselines/dreamerv3_official/full26_size12m"
HTS_V19 = ROOT / "logs/external_baselines/dreamerv3_official_hts_v19_ab/hts_locked_hier_x3"
HTS_V20 = ROOT / "logs/external_baselines/dreamerv3_official_hts_v20_fair_compare/hts_locked_hier_x3"
NO_VC = ROOT / "logs/external_baselines/dreamerv3_official_hts_no_vc/hts_locked_hier_x3_no_vc"

GAMES = ["alien", "breakout"]
SEEDS = [0, 1, 2, 3, 4]
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


def add_record(method, condition, game, seed, source, logdir):
  rec = {
      "method": method,
      "condition": condition,
      "game": game,
      "seed": seed,
      "source": source,
      "logdir": str(logdir),
      "metric_source_files": f"{logdir / 'scores.jsonl'};{logdir / 'paper_artifacts/episode_scores.jsonl'}",
  }
  rec.update(metrics_for_rows(read_scores(logdir)))
  rec["included_in_completed_primary"] = bool(rec["completed"])
  rec["exclusion_reason"] = "" if rec["completed"] else "missing_or_partial_final_bin"
  return rec


def dreamer_record(game, seed):
  candidates = []
  for name, source in [
      (f"seed_{seed}", "official_full26_original"),
      (f"seed_{seed}_repair_no_video", "official_full26_repair_no_video"),
      (f"seed_{seed}_repair2_no_video", "official_full26_repair2_no_video"),
  ]:
    logdir = DREAMER / game / name
    if logdir.exists():
      candidates.append(add_record("DreamerV3", "dreamer_v3_reference", game, seed, source, logdir))
  if not candidates:
    return add_record("DreamerV3", "dreamer_v3_reference", game, seed, "missing", DREAMER / game / f"seed_{seed}")
  completed = [x for x in candidates if x["completed"]]
  if completed:
    completed.sort(key=lambda x: (not x["source"].endswith("original"), -x["latest_raw_frames"]))
    return completed[0]
  candidates.sort(key=lambda x: x["latest_raw_frames"], reverse=True)
  return candidates[0]


def hts_v20_record(game, seed):
  if seed >= 3:
    logdir = HTS_V20 / game / f"seed_{seed}"
    return add_record("HTS", "v20_locked_hier_x3", game, seed, "V20_fair_compare", logdir)
  candidates = []
  for name, source in [
      (f"seed_{seed}", "V19_original"),
      (f"seed_{seed}_retry_no_video", "V19_retry_no_video"),
  ]:
    logdir = HTS_V19 / game / name
    if logdir.exists():
      candidates.append(add_record("HTS", "v20_locked_hier_x3", game, seed, source, logdir))
  if not candidates:
    return add_record("HTS", "v20_locked_hier_x3", game, seed, "missing", HTS_V19 / game / f"seed_{seed}")
  completed = [x for x in candidates if x["completed"]]
  if completed:
    completed.sort(key=lambda x: ("retry" not in x["source"], -x["latest_raw_frames"]))
    return completed[0]
  candidates.sort(key=lambda x: x["latest_raw_frames"], reverse=True)
  return candidates[0]


def no_vc_record(game, seed):
  return add_record(
      "HTS",
      "no_vc_locked_hier_x3",
      game,
      seed,
      "no_vc_probe",
      NO_VC / game / f"seed_{seed}")


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
  cols = list(rows[0].keys()) if rows else []
  lines = []
  if cols:
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
      lines.append("| " + " | ".join(str(clean(row.get(col, ""))) for col in cols) + " |")
  path.write_text("\n".join(lines) + ("\n" if lines else "No rows.\n"))


def mean(values):
  vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
  return float(statistics.mean(vals)) if vals else math.nan


def aggregate(game, condition, rows):
  completed = [r for r in rows if r["game"] == game and r["condition"] == condition and r["completed"]]
  return {
      "game": game,
      "condition": condition,
      "completed_seeds": ",".join(str(r["seed"]) for r in completed),
      "num_completed": len(completed),
      "auc_20bin_mean": mean([r["auc_20bin_mean"] for r in completed]),
      "final_20pct_mean": mean([r["final_20pct_mean"] for r in completed]),
      "final_bin_mean": mean([r["final_bin_mean"] for r in completed]),
      "latest_episode_score_mean": mean([r["latest_episode_score"] for r in completed]),
  }


def add_deltas(rows):
  by = {(r["game"], r["condition"]): r for r in rows}
  for row in rows:
    if row["condition"] != "no_vc_locked_hier_x3":
      row["delta_auc_vs_v20_hts"] = ""
      row["delta_final_20pct_vs_v20_hts"] = ""
      row["delta_auc_vs_dreamer"] = ""
      row["delta_final_20pct_vs_dreamer"] = ""
      continue
    v20 = by.get((row["game"], "v20_locked_hier_x3"), {})
    dreamer = by.get((row["game"], "dreamer_v3_reference"), {})
    row["delta_auc_vs_v20_hts"] = row["auc_20bin_mean"] - v20.get("auc_20bin_mean", math.nan)
    row["delta_final_20pct_vs_v20_hts"] = row["final_20pct_mean"] - v20.get("final_20pct_mean", math.nan)
    row["delta_auc_vs_dreamer"] = row["auc_20bin_mean"] - dreamer.get("auc_20bin_mean", math.nan)
    row["delta_final_20pct_vs_dreamer"] = row["final_20pct_mean"] - dreamer.get("final_20pct_mean", math.nan)


def plot(records):
  centers = (BINS[:-1] + BINS[1:]) / 2
  colors = {
      "dreamer_v3_reference": "#4C78A8",
      "v20_locked_hier_x3": "#F58518",
      "no_vc_locked_hier_x3": "#54A24B",
  }
  labels = {
      "dreamer_v3_reference": "DreamerV3",
      "v20_locked_hier_x3": "HTS V20",
      "no_vc_locked_hier_x3": "HTS no VC",
  }
  fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
  for ax, game in zip(axes, GAMES):
    for condition in ["dreamer_v3_reference", "v20_locked_hier_x3", "no_vc_locked_hier_x3"]:
      rows = [r for r in records if r["game"] == game and r["condition"] == condition and r["completed"]]
      if not rows:
        continue
      arr = np.array([r["bin_means"] for r in rows], dtype=float)
      m = np.nanmean(arr, axis=0)
      n = np.sum(np.isfinite(arr), axis=0)
      sem = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
      ax.plot(centers, m, lw=2.2, color=colors[condition], label=f"{labels[condition]} (n={len(rows)})")
      ax.fill_between(centers, m - sem, m + sem, color=colors[condition], alpha=0.16, linewidth=0)
    ax.set_title(game.title())
    ax.set_xlabel("Raw frames")
    ax.set_xlim(0, TARGET_FRAMES)
    ax.set_xticks([200000, 400000])
    ax.set_xticklabels(["200K", "400K"])
    ax.grid(alpha=0.25)
  axes[0].set_ylabel("Episode score")
  axes[1].legend(fontsize=8, loc="best")
  fig.tight_layout()
  fig.savefig(OUT / "fig_no_vc_alien_breakout_20bin_curves.png", dpi=180)
  fig.savefig(OUT / "fig_no_vc_alien_breakout_20bin_curves.pdf")


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  records = []
  for game in GAMES:
    for seed in SEEDS:
      records.append(dreamer_record(game, seed))
      records.append(hts_v20_record(game, seed))
      records.append(no_vc_record(game, seed))
  write_json(OUT / "no_vc_raw_metrics.json", records)
  write_csv(OUT / "no_vc_raw_metrics.csv", records)
  write_md_table(OUT / "no_vc_raw_metrics.md", records)
  aggs = []
  for game in GAMES:
    for condition in ["dreamer_v3_reference", "v20_locked_hier_x3", "no_vc_locked_hier_x3"]:
      aggs.append(aggregate(game, condition, records))
  add_deltas(aggs)
  write_json(OUT / "no_vc_aggregate.json", aggs)
  write_csv(OUT / "no_vc_aggregate.csv", aggs)
  write_md_table(OUT / "no_vc_aggregate.md", aggs)
  plot(records)
  decision = {
      "all_no_vc_complete": all(
          r["num_completed"] == len(SEEDS)
          for r in aggs if r["condition"] == "no_vc_locked_hier_x3"),
      "protocol": "20 bins over raw frames 0..440000; AUC is mean of bin means; final-20pct is mean of last 4 bins; final-bin is last bin.",
      "aggregate_rows": aggs,
  }
  write_json(OUT / "no_vc_decision.json", decision)
  print(json.dumps(decision, indent=2, sort_keys=True, default=clean))


if __name__ == "__main__":
  main()
