#!/usr/bin/env python3
"""Replay local DreamerV3 full26 Atari100K logs into W&B.

This does not train. It reads the selected local full26 DreamerV3 result
manifest and uploads one W&B run per game/seed. By default, each game gets a
separate project:

  <project-prefix>-<game>

The uploaded curves use raw Atari environment frames as the W&B step.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/disk1/backup_user/dat.tt2/xuance")
ARTIFACT_DIR = (
    ROOT
    / "external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_final"
)
DEFAULT_SELECTED = ARTIFACT_DIR / "full26_selected_seed_metrics.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  if not path.exists():
    return rows
  with path.open() as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        rows.append(json.loads(line))
      except json.JSONDecodeError:
        continue
  return rows


def finite(value: Any) -> bool:
  return isinstance(value, (int, float)) and math.isfinite(float(value))


def clean_metric_dict(row: dict[str, Any], prefix: str = "") -> dict[str, float]:
  out = {}
  for key, value in row.items():
    if finite(value):
      out[f"{prefix}{key}"] = float(value)
  return out


def selected_rows(path: Path, games: set[str] | None, seeds: set[int] | None):
  data = json.loads(path.read_text())
  rows = [x for x in data if x.get("selected") and x.get("completed")]
  if games:
    rows = [x for x in rows if str(x.get("game")) in games]
  if seeds:
    rows = [x for x in rows if int(x.get("seed")) in seeds]
  rows.sort(key=lambda x: (str(x["game"]), int(x["seed"])))
  return rows


def project_name(args, game: str) -> str:
  if args.project_mode == "single":
    return args.project
  return f"{args.project_prefix}-{game}"


def run_id(args, game: str, seed: int) -> str:
  safe = lambda x: str(x).replace("/", "_").replace(" ", "_")
  return safe(f"{args.dataset}-{args.method_slug}-{game}-seed{seed}-{args.upload_tag}")


def display_name(args, seed: int) -> str:
  if args.method_slug == "dreamerv3-official-size12m":
    return f"DreamerV3-size12m-seed{seed}"
  return f"{args.method_slug}-seed{seed}"


def upload_run(args, row: dict[str, Any], dry_run: bool):
  import wandb

  game = str(row["game"])
  seed = int(row["seed"])
  logdir = Path(row["logdir"])
  metrics_path = logdir / "metrics.jsonl"
  scores_path = logdir / "scores.jsonl"
  source_path = metrics_path if metrics_path.exists() else scores_path
  event_rows = load_jsonl(source_path)

  project = project_name(args, game)
  name = display_name(args, seed)
  group = f"{args.dataset}/{game}"
  tags = [
      args.dataset,
      args.method_slug,
      "official_dreamerv3",
      "full26",
      game,
      f"seed{seed}",
      "local_replay_upload",
      args.upload_tag,
  ]

  config = {
      "method": row.get("method", "DreamerV3"),
      "method_slug": args.method_slug,
      "dataset": args.dataset,
      "game": game,
      "seed": seed,
      "source": "local_jsonl_replay",
      "source_logdir": str(logdir),
      "source_metrics_file": str(source_path),
      "selection_reason": row.get("selection_reason"),
      "model_size_config": "size12m",
      "task": f"atari100k_{game}",
      "raw_frame_budget": 400000,
      "action_repeat": 4,
      "agent_step_budget": 100000,
      "train_ratio": 256,
      "batch_size": 16,
      "batch_length": 64,
      "upload_tag": args.upload_tag,
  }

  summary = {
      "summary/latest_raw_frames": row.get("latest_raw_frames"),
      "summary/latest_episode_score": row.get("latest_episode_score"),
      "summary/final_20pct_mean": row.get("final_20pct_mean"),
      "summary/final_bin_mean": row.get("final_bin_mean"),
      "summary/auc_20bin_mean": row.get("auc_20bin_mean"),
      "summary/best_bin_score": row.get("best_bin_score"),
      "summary/episode_score_rows": row.get("episode_score_rows"),
      "summary/completed": bool(row.get("completed")),
  }

  if dry_run:
    return {
        "project": project,
        "name": name,
        "id": run_id(args, game, seed),
        "source": str(source_path),
        "events": len(event_rows),
        "summary": summary,
    }

  run = wandb.init(
      project=project,
      name=name,
      id=run_id(args, game, seed),
      resume=args.resume,
      group=group,
      job_type="local_replay_upload",
      tags=tags,
      config=config,
      dir=str(args.wandb_dir),
      reinit=True,
  )

  for key, value in summary.items():
    if finite(value) or isinstance(value, bool):
      run.summary[key] = value

  for event in event_rows:
    if "step" not in event:
      continue
    step = int(float(event["step"]))
    payload = clean_metric_dict(event)
    if "episode/score" in payload:
      payload["episode_score"] = payload["episode/score"]
    payload["raw_frames"] = float(step)
    payload["agent_actions_est"] = float(step) / 4.0
    run.log(payload, step=step)

  bin_means = row.get("bin_means") or []
  bin_counts = row.get("bin_counts") or []
  for idx, mean in enumerate(bin_means):
    if not finite(mean):
      continue
    step = int((idx + 0.5) * 400000 / max(len(bin_means), 1))
    payload = {
        "curve_20bin/score": float(mean),
        "curve_20bin/bin_index": idx,
        "curve_20bin/bin_count": float(bin_counts[idx]) if idx < len(bin_counts) else 0.0,
        "raw_frames": float(step),
        "agent_actions_est": float(step) / 4.0,
    }
    run.log(payload, step=step)

  for path in [scores_path, metrics_path]:
    if path.exists() and args.save_source_files:
      run.save(str(path), policy="now")

  curve_png = ARTIFACT_DIR / "per_game_curves" / f"{game}.png"
  if curve_png.exists() and args.log_game_curve_image:
    run.log({"game_curve_20bin": wandb.Image(str(curve_png))})

  run.finish()
  return {
      "project": project,
      "name": name,
      "id": run_id(args, game, seed),
      "source": str(source_path),
      "events": len(event_rows),
      "uploaded": True,
  }


def write_manifest(args, manifest: list[dict[str, Any]]):
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True))
  md = args.output.with_suffix(".md")
  lines = [
      "# W&B Upload Manifest",
      "",
      f"- dry_run: `{args.dry_run}`",
      f"- project_mode: `{args.project_mode}`",
      f"- dataset: `{args.dataset}`",
      f"- runs: `{len(manifest)}`",
      "",
      "| project | run | id | events | source |",
      "|---|---|---:|---:|---|",
  ]
  for item in manifest:
    lines.append(
        f"| {item['project']} | {item['name']} | {item['id']} | "
        f"{item['events']} | {item['source']} |"
    )
  md.write_text("\n".join(lines) + "\n")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
  parser.add_argument("--games", default="", help="Comma-separated games; empty means all.")
  parser.add_argument("--seeds", default="", help="Comma-separated seeds; empty means all.")
  parser.add_argument("--dataset", default="atari100k")
  parser.add_argument("--method-slug", default="dreamerv3-official-size12m")
  parser.add_argument("--project-mode", choices=["per_game", "single"], default="per_game")
  parser.add_argument("--project-prefix", default="dreamv3")
  parser.add_argument("--project", default="dreamv3-atari100k-full26")
  parser.add_argument("--upload-tag", default="local-replay-v1")
  parser.add_argument("--resume", choices=["allow", "must", "never", "auto"], default="allow")
  parser.add_argument("--wandb-dir", type=Path, default=ROOT / "wandb_uploads")
  parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "wandb_upload_manifest.json")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--save-source-files", action="store_true")
  parser.add_argument("--log-game-curve-image", action="store_true")
  args = parser.parse_args()

  games = {x.strip() for x in args.games.split(",") if x.strip()}
  seeds = {int(x.strip()) for x in args.seeds.split(",") if x.strip()}
  rows = selected_rows(args.selected, games or None, seeds or None)
  if not rows:
    raise SystemExit("No selected completed rows found.")

  if not args.dry_run:
    try:
      import wandb  # noqa: F401
    except ImportError as exc:
      raise SystemExit("wandb is not installed in this Python environment.") from exc

  manifest = []
  for row in rows:
    manifest.append(upload_run(args, row, args.dry_run))
  write_manifest(args, manifest)
  print(f"Wrote manifest: {args.output}")
  print(f"Runs: {len(manifest)}")


if __name__ == "__main__":
  main()
