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
LOGROOT = ROOT / "logs/external_baselines/dreamerv3_official/full26_size12m"
OUT = ROOT / "external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_final"

GAMES = [
    "alien", "amidar", "assault", "asterix", "bank_heist", "battle_zone",
    "boxing", "breakout", "chopper_command", "crazy_climber", "demon_attack",
    "freeway", "frostbite", "gopher", "hero", "james_bond", "kangaroo",
    "krull", "kung_fu_master", "ms_pacman", "pong", "private_eye", "qbert",
    "road_runner", "seaquest", "up_n_down",
]
SEEDS = [0, 1, 2, 3, 4]
TARGET_FRAMES = 440000
BINS = np.linspace(0, TARGET_FRAMES, 21)
COMPLETED_THRESHOLD = BINS[-2]


def read_scores(logdir):
  path = pathlib.Path(logdir) / "scores.jsonl"
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
    if "step" not in item or "episode/score" not in item:
      continue
    try:
      rows.append({
          "raw_frames": float(item["step"]),
          "episode_score": float(item["episode/score"]),
      })
    except Exception:
      continue
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
      "episode_score_rows": len(rows),
      "completed": completed,
      "auc_20bin_mean": float(np.nanmean(arr)) if np.isfinite(arr).any() else math.nan,
      "final_20pct_mean": float(np.nanmean(arr[16:])) if np.isfinite(arr[16:]).any() else math.nan,
      "final_bin_mean": float(arr[-1]) if not math.isnan(arr[-1]) else math.nan,
      "latest_episode_score": float(ys[-1]),
      "best_bin_score": float(np.nanmax(arr)) if np.isfinite(arr).any() else math.nan,
      "bin_means": bin_means,
      "bin_counts": bin_counts,
  }


def record_for_logdir(game, seed, logdir):
  rec = {
      "method": "DreamerV3",
      "game": game,
      "seed": seed,
      "run_dir": pathlib.Path(logdir).name,
      "logdir": str(logdir),
      "metric_source_files": str(pathlib.Path(logdir) / "scores.jsonl"),
  }
  rec.update(metrics_for_rows(read_scores(logdir)))
  return rec


def select_record(game, seed):
  game_dir = LOGROOT / game
  candidates = []
  for logdir in sorted(game_dir.glob(f"seed_{seed}*")):
    if logdir.is_dir():
      candidates.append(record_for_logdir(game, seed, logdir))
  if not candidates:
    rec = record_for_logdir(game, seed, game_dir / f"seed_{seed}")
    rec["selection_reason"] = "missing"
    return rec, []
  completed = [c for c in candidates if c["completed"]]
  if completed:
    # Prefer completed repair runs over failed originals, but prefer original
    # when both original and repair are complete and similarly advanced.
    completed.sort(key=lambda r: (r["run_dir"] != f"seed_{seed}", -r["latest_raw_frames"]))
    selected = completed[0]
    selected["selection_reason"] = "completed_original_preferred" if selected["run_dir"] == f"seed_{seed}" else "completed_repair_selected"
    return selected, candidates
  candidates.sort(key=lambda r: r["latest_raw_frames"], reverse=True)
  selected = candidates[0]
  selected["selection_reason"] = "partial_best_latest_frames"
  return selected, candidates


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


def aggregate(records):
  rows = []
  for game in GAMES:
    rs = [r for r in records if r["game"] == game]
    completed = [r for r in rs if r["completed"]]
    rows.append({
        "game": game,
        "num_selected_seeds": len(rs),
        "completed_seeds": ",".join(str(r["seed"]) for r in completed),
        "num_completed": len(completed),
        "missing_or_incomplete_seeds": ",".join(str(r["seed"]) for r in rs if not r["completed"]),
        "auc_20bin_mean": mean([r["auc_20bin_mean"] for r in completed]),
        "final_20pct_mean": mean([r["final_20pct_mean"] for r in completed]),
        "final_bin_mean": mean([r["final_bin_mean"] for r in completed]),
        "latest_episode_score_mean": mean([r["latest_episode_score"] for r in completed]),
        "latest_raw_frames_min": min([r["latest_raw_frames"] for r in rs]) if rs else 0,
        "latest_raw_frames_max": max([r["latest_raw_frames"] for r in rs]) if rs else 0,
    })
  return rows


def write_normalized_curves(records):
  rows = []
  centers = (BINS[:-1] + BINS[1:]) / 2
  for game in GAMES:
    rs = [r for r in records if r["game"] == game and r["completed"]]
    if not rs:
      continue
    arr = np.array([r["bin_means"] for r in rs], dtype=float)
    means = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    n = np.sum(np.isfinite(arr), axis=0)
    sem = std / np.sqrt(np.maximum(n, 1))
    for idx, (center, m, s, count) in enumerate(zip(centers, means, sem, n)):
      rows.append({
          "game": game,
          "bin_index": idx,
          "raw_frame_center": float(center),
          "score_mean": float(m) if not math.isnan(m) else math.nan,
          "score_sem": float(s) if not math.isnan(s) else math.nan,
          "num_seed_values": int(count),
      })
  write_csv(OUT / "full26_learning_curves_20bin.csv", rows)
  write_json(OUT / "full26_learning_curves_20bin.json", rows)


def plot_grid(records):
  centers = (BINS[:-1] + BINS[1:]) / 2
  fig, axes = plt.subplots(5, 6, figsize=(22, 16), sharex=True)
  axes = axes.ravel()
  for ax, game in zip(axes, GAMES):
    rs = [r for r in records if r["game"] == game and r["completed"]]
    if rs:
      arr = np.array([r["bin_means"] for r in rs], dtype=float)
      m = np.nanmean(arr, axis=0)
      n = np.sum(np.isfinite(arr), axis=0)
      sem = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
      ax.plot(centers, m, color="#4C78A8", lw=2.0)
      ax.fill_between(centers, m - sem, m + sem, color="#4C78A8", alpha=0.18, linewidth=0)
      ax.text(0.02, 0.88, f"n={len(rs)}", transform=ax.transAxes, fontsize=8)
    else:
      ax.text(0.5, 0.5, "missing", ha="center", va="center")
    ax.set_title(game.replace("_", " ").title(), fontsize=10)
    ax.set_xlim(0, TARGET_FRAMES)
    ax.set_xticks([200000, 400000])
    ax.set_xticklabels(["200K", "400K"])
    ax.grid(alpha=0.22)
  for ax in axes[len(GAMES):]:
    ax.axis("off")
  fig.supxlabel("Raw frames")
  fig.supylabel("Episode score")
  fig.tight_layout()
  fig.savefig(OUT / "fig_dreamerv3_full26_learning_curves_20bin.png", dpi=180)
  fig.savefig(OUT / "fig_dreamerv3_full26_learning_curves_20bin.pdf")


def plot_individual(records):
  centers = (BINS[:-1] + BINS[1:]) / 2
  imgdir = OUT / "per_game_curves"
  imgdir.mkdir(exist_ok=True)
  for game in GAMES:
    rs = [r for r in records if r["game"] == game and r["completed"]]
    fig, ax = plt.subplots(figsize=(5, 4))
    for r in rs:
      ax.plot(centers, r["bin_means"], color="#B7C9E2", lw=1.0, alpha=0.65)
    if rs:
      arr = np.array([r["bin_means"] for r in rs], dtype=float)
      m = np.nanmean(arr, axis=0)
      n = np.sum(np.isfinite(arr), axis=0)
      sem = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
      ax.plot(centers, m, color="#4C78A8", lw=2.4, label=f"mean (n={len(rs)})")
      ax.fill_between(centers, m - sem, m + sem, color="#4C78A8", alpha=0.18, linewidth=0)
      ax.legend(fontsize=8)
    ax.set_title(game.replace("_", " ").title())
    ax.set_xlabel("Raw frames")
    ax.set_ylabel("Episode score")
    ax.set_xlim(0, TARGET_FRAMES)
    ax.set_xticks([200000, 400000])
    ax.set_xticklabels(["200K", "400K"])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(imgdir / f"{game}.png", dpi=180)
    fig.savefig(imgdir / f"{game}.pdf")
    plt.close(fig)


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  selected, candidates = [], []
  for game in GAMES:
    for seed in SEEDS:
      rec, cands = select_record(game, seed)
      selected.append(rec)
      for cand in cands:
        cand["selected"] = cand["logdir"] == rec["logdir"]
        cand["selected_run_dir"] = rec["run_dir"]
      candidates.extend(cands)

  summary = aggregate(selected)
  missing_games = [row["game"] for row in summary if row["num_completed"] != len(SEEDS)]
  incomplete = [r for r in selected if not r["completed"]]
  repair_selected = [r for r in selected if "repair" in r["run_dir"]]
  report = {
      "logroot": str(LOGROOT),
      "target_games": len(GAMES),
      "target_seeds_per_game": len(SEEDS),
      "target_runs": len(GAMES) * len(SEEDS),
      "selected_runs": len(selected),
      "completed_selected_runs": sum(1 for r in selected if r["completed"]),
      "all_games_complete": not missing_games,
      "games_with_missing_or_incomplete_selected_runs": missing_games,
      "repair_selected_runs": [{"game": r["game"], "seed": r["seed"], "run_dir": r["run_dir"]} for r in repair_selected],
      "metric_protocol": {
          "source": "scores.jsonl episode/score",
          "x_axis": "raw environment frames",
          "bins": 20,
          "target_frames": TARGET_FRAMES,
          "completed_threshold": float(COMPLETED_THRESHOLD),
          "primary_curve": "mean over completed seeds per fixed raw-frame bin; shaded SEM",
      },
  }

  write_json(OUT / "full26_completion_report.json", report)
  write_json(OUT / "full26_selected_seed_metrics.json", selected)
  write_json(OUT / "full26_all_candidate_runs.json", candidates)
  write_csv(OUT / "full26_selected_seed_metrics.csv", selected)
  write_csv(OUT / "full26_summary_by_game.csv", summary)
  write_md_table(OUT / "full26_summary_by_game.md", summary)
  write_normalized_curves(selected)
  plot_grid(selected)
  plot_individual(selected)

  lines = [
      "# DreamerV3 Full26 Final Audit",
      "",
      f"Log root: `{LOGROOT}`",
      "",
      f"Selected runs: `{report['selected_runs']}` / `{report['target_runs']}`",
      f"Completed selected runs: `{report['completed_selected_runs']}` / `{report['target_runs']}`",
      f"All games complete: `{report['all_games_complete']}`",
      "",
      "## Missing Or Incomplete",
      "",
      "`" + ",".join(missing_games) + "`" if missing_games else "None.",
      "",
      "## Repair Runs Selected",
      "",
  ]
  if repair_selected:
    lines += [f"- `{r['game']}` seed `{r['seed']}` -> `{r['run_dir']}`" for r in repair_selected]
  else:
    lines.append("None.")
  lines += [
      "",
      "## Outputs",
      "",
      "- `fig_dreamerv3_full26_learning_curves_20bin.png/pdf`",
      "- `per_game_curves/*.png/pdf`",
      "- `full26_learning_curves_20bin.csv/json`",
      "- `full26_selected_seed_metrics.csv/json`",
      "- `full26_summary_by_game.csv/md`",
  ]
  (OUT / "full26_audit_report.md").write_text("\n".join(lines) + "\n")
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
