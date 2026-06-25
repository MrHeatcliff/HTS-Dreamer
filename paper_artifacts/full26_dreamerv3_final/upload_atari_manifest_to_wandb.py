#!/usr/bin/env python3
"""Upload Atari local-result manifests to W&B with matched metrics.

This is the generic companion to the DreamerV3 full26 uploader. It can replay
HTS, ablations, and sweeps into the same per-game projects as DreamerV3:

  dreamv3-<game>

Metric naming intentionally matches upload_full26_dreamerv3_to_wandb.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/disk1/backup_user/dat.tt2/xuance")
ARTIFACT_DIR = (
    ROOT
    / "external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_final"
)


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


def split_csv(value: str) -> set[str]:
  return {x.strip() for x in value.split(",") if x.strip()}


def safe_token(value: Any) -> str:
  out = str(value).strip().replace("/", "_").replace(" ", "_")
  return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in out)


def load_manifest(path: Path) -> list[dict[str, Any]]:
  data = json.loads(path.read_text())
  if isinstance(data, list):
    return data
  if isinstance(data, dict):
    for key in ("rows", "records", "metrics", "runs"):
      if isinstance(data.get(key), list):
        return data[key]
  raise ValueError(f"Unsupported manifest structure: {path}")


def normalize_condition(row: dict[str, Any], default: str, force: bool = False) -> str:
  if force:
    return str(default)
  for key in ("condition", "variant", "source"):
    value = row.get(key)
    if value not in (None, ""):
      cond = str(value)
      break
  else:
    method = str(row.get("method", default))
    cond = method.lower().replace(" ", "_")
  if row.get("cov_scale") not in (None, ""):
    cond = f"{cond}_cov{safe_token(row.get('cov_scale'))}"
  if row.get("levels") not in (None, ""):
    cond = f"{cond}_levels{safe_token(row.get('levels'))}"
  if row.get("config_levels") not in (None, "") and row.get("levels") in (None, ""):
    cond = f"{cond}_levels{safe_token(row.get('config_levels'))}"
  if row.get("warmup_raw") not in (None, ""):
    cond = f"{cond}_warmup{safe_token(row.get('warmup_raw'))}"
  return cond


def selected_rows(args) -> list[dict[str, Any]]:
  rows = load_manifest(args.manifest)
  games = split_csv(args.games)
  seeds = {int(x) for x in split_csv(args.seeds)}
  methods = split_csv(args.methods)
  conditions = split_csv(args.conditions)
  exclude_conditions = split_csv(args.exclude_conditions)

  out = []
  for row in rows:
    if args.completed_only and not bool(row.get("completed", False)):
      continue
    if args.primary_only and row.get("included_in_completed_primary") is False:
      continue
    game = str(row.get("game", ""))
    if games and game not in games:
      continue
    if seeds and int(row.get("seed", -1)) not in seeds:
      continue
    method = str(row.get("method", args.method_slug))
    if methods and method not in methods:
      continue
    condition = normalize_condition(row, args.condition, args.force_condition)
    if conditions and condition not in conditions:
      continue
    if exclude_conditions and condition in exclude_conditions:
      continue
    row = dict(row)
    row["_condition"] = condition
    row["_method"] = method
    out.append(row)
  out.sort(key=lambda x: (str(x.get("game")), str(x.get("_condition")), int(x.get("seed", -1))))
  return out


def project_name(args, game: str) -> str:
  if args.project_mode == "single":
    return args.project
  return f"{args.project_prefix}-{game}"


def run_id(args, row: dict[str, Any]) -> str:
  if args.id_from_logdir:
    digest = hashlib.sha1(str(row.get("logdir", "")).encode()).hexdigest()[:10]
  else:
    digest = args.upload_tag
  return safe_token(
      f"{args.dataset}-{args.method_slug}-{row.get('game')}-"
      f"{row.get('_condition')}-seed{row.get('seed')}-{digest}")


def metric_file(row: dict[str, Any]) -> Path:
  logdir = Path(str(row["logdir"]))
  metrics = logdir / "metrics.jsonl"
  scores = logdir / "scores.jsonl"
  return metrics if metrics.exists() else scores


def display_name(args, row: dict[str, Any]) -> str:
  seed = int(row["seed"])
  condition = str(row["_condition"])
  method_slug = str(args.method_slug)

  if method_slug == "hts-v20-locked-hier-x3" or condition == "v20_locked_hier_x3":
    return f"HTS-main-hier0.3-seed{seed}"
  if method_slug == "hts-no-vc" or condition == "no_vc_locked_hier_x3":
    return f"HTS-noVC-seed{seed}"
  if method_slug == "hts-new-v22":
    label = str(row.get("display_condition") or condition)
    if label.startswith("new-"):
      label = label[len("new-"):]
    return f"HTS-new-{safe_token(label)}-seed{seed}"
  if "cov_sweep_cov" in condition:
    value = condition.split("cov_sweep_cov", 1)[1]
    return f"HTS-cov{value}-seed{seed}"
  if "levels" in condition:
    suffix = condition.rsplit("levels", 1)[1]
    return f"HTS-L{suffix}-seed{seed}"
  if method_slug == "hts-v21-warmup":
    if condition.endswith("_raw"):
      return f"HTS-warmup{condition.replace('_raw', '')}-seed{seed}"
    return f"HTS-warmup-{safe_token(condition)}-seed{seed}"
  return f"{safe_token(method_slug)}-{safe_token(condition)}-seed{seed}"


def upload_run(args, row: dict[str, Any], dry_run: bool):
  import wandb

  game = str(row["game"])
  seed = int(row["seed"])
  condition = str(row["_condition"])
  method = str(row["_method"])
  logdir = Path(str(row["logdir"]))
  source_path = metric_file(row)
  event_rows = load_jsonl(source_path)

  project = project_name(args, game)
  method_slug = safe_token(args.method_slug)
  condition_slug = safe_token(condition)
  name = display_name(args, row)
  group = f"{args.dataset}/{game}/{method_slug}/{condition_slug}"
  tags = [
      args.dataset,
      method_slug,
      condition_slug,
      game,
      f"seed{seed}",
      "local_replay_upload",
      args.upload_tag,
  ]

  config = {
      "method": method,
      "method_slug": args.method_slug,
      "condition": condition,
      "dataset": args.dataset,
      "game": game,
      "seed": seed,
      "source": "local_jsonl_replay",
      "source_logdir": str(logdir),
      "source_metrics_file": str(source_path),
      "source_manifest": str(args.manifest),
      "raw_frame_budget": args.raw_frame_budget,
      "action_repeat": args.action_repeat,
      "agent_step_budget": args.raw_frame_budget / args.action_repeat,
      "train_ratio": args.train_ratio,
      "batch_size": args.batch_size,
      "batch_length": args.batch_length,
      "upload_tag": args.upload_tag,
  }
  for key, value in row.items():
    if key.startswith("_"):
      continue
    if key in config:
      continue
    if finite(value) or isinstance(value, (str, bool, int, float)):
      config[f"manifest/{key}"] = value

  summary = {
      "summary/latest_raw_frames": row.get("latest_raw_frames"),
      "summary/latest_episode_score": row.get("latest_episode_score"),
      "summary/final_20pct_mean": row.get("final_20pct_mean"),
      "summary/final_bin_mean": row.get("final_bin_mean"),
      "summary/auc_20bin_mean": row.get("auc_20bin_mean"),
      "summary/best_bin_score": row.get("best_bin_score"),
      "summary/episode_score_rows": row.get("episode_score_rows"),
      "summary/completed": bool(row.get("completed")),
      "summary/condition": condition,
      "summary/method_slug": args.method_slug,
  }

  if dry_run:
    return {
        "project": project,
        "name": name,
        "id": run_id(args, row),
        "method": method,
        "condition": condition,
        "game": game,
        "seed": seed,
        "source": str(source_path),
        "events": len(event_rows),
        "summary": summary,
    }

  run = wandb.init(
      project=project,
      name=name,
      id=run_id(args, row),
      resume=args.resume,
      group=group,
      job_type="local_replay_upload",
      tags=tags,
      config=config,
      dir=str(args.wandb_dir),
      reinit=True,
  )

  for key, value in summary.items():
    if finite(value) or isinstance(value, (bool, str)):
      run.summary[key] = value

  for event in event_rows:
    if "step" not in event:
      continue
    step = int(float(event["step"]))
    payload = clean_metric_dict(event)
    if "episode/score" in payload:
      payload["episode_score"] = payload["episode/score"]
    payload["raw_frames"] = float(step)
    payload["agent_actions_est"] = float(step) / float(args.action_repeat)
    payload["upload/method_index"] = float(args.method_index)
    run.log(payload, step=step)

  bin_means = row.get("bin_means") or []
  bin_counts = row.get("bin_counts") or []
  for idx, mean in enumerate(bin_means):
    if not finite(mean):
      continue
    step = int((idx + 0.5) * args.raw_frame_budget / max(len(bin_means), 1))
    payload = {
        "curve_20bin/score": float(mean),
        "curve_20bin/bin_index": idx,
        "curve_20bin/bin_count": float(bin_counts[idx]) if idx < len(bin_counts) else 0.0,
        "raw_frames": float(step),
        "agent_actions_est": float(step) / float(args.action_repeat),
        "upload/method_index": float(args.method_index),
    }
    run.log(payload, step=step)

  if args.save_source_files:
    for path in [logdir / "scores.jsonl", logdir / "metrics.jsonl"]:
      if path.exists():
        run.save(str(path), policy="now")

  run.finish()
  return {
      "project": project,
      "name": name,
      "id": run_id(args, row),
      "method": method,
      "condition": condition,
      "game": game,
      "seed": seed,
      "source": str(source_path),
      "events": len(event_rows),
      "uploaded": True,
  }


def write_manifest(args, manifest: list[dict[str, Any]]):
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True))
  md = args.output.with_suffix(".md")
  lines = [
      "# W&B Atari Upload Manifest",
      "",
      f"- dry_run: `{args.dry_run}`",
      f"- source_manifest: `{args.manifest}`",
      f"- project_mode: `{args.project_mode}`",
      f"- runs: `{len(manifest)}`",
      "",
      "| project | run | method | condition | game | seed | events | source |",
      "|---|---|---|---|---|---:|---:|---|",
  ]
  for item in manifest:
    lines.append(
        f"| {item['project']} | {item['name']} | {item['method']} | "
        f"{item['condition']} | {item['game']} | {item['seed']} | "
        f"{item['events']} | {item['source']} |"
    )
  md.write_text("\n".join(lines) + "\n")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--games", default="")
  parser.add_argument("--seeds", default="")
  parser.add_argument("--methods", default="")
  parser.add_argument("--conditions", default="")
  parser.add_argument("--exclude-conditions", default="")
  parser.add_argument("--completed-only", action="store_true", default=True)
  parser.add_argument("--no-completed-only", dest="completed_only", action="store_false")
  parser.add_argument("--primary-only", action="store_true")
  parser.add_argument("--dataset", default="atari100k")
  parser.add_argument("--method-slug", default="hts")
  parser.add_argument("--condition", default="hts")
  parser.add_argument("--force-condition", action="store_true")
  parser.add_argument("--method-index", type=float, default=1.0)
  parser.add_argument("--project-mode", choices=["per_game", "single"], default="per_game")
  parser.add_argument("--project-prefix", default="dreamv3")
  parser.add_argument("--project", default="dreamv3-atari100k-full26")
  parser.add_argument("--upload-tag", default="local-replay-v1")
  parser.add_argument("--id-from-logdir", action="store_true", default=True)
  parser.add_argument("--resume", choices=["allow", "must", "never", "auto"], default="allow")
  parser.add_argument("--wandb-dir", type=Path, default=ROOT / "wandb_uploads")
  parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "wandb_atari_upload_manifest.json")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--save-source-files", action="store_true")
  parser.add_argument("--raw-frame-budget", type=int, default=400000)
  parser.add_argument("--action-repeat", type=int, default=4)
  parser.add_argument("--train-ratio", type=int, default=256)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--batch-length", type=int, default=64)
  args = parser.parse_args()

  rows = selected_rows(args)
  if not rows:
    raise SystemExit("No rows matched upload filters.")

  if not args.dry_run:
    try:
      import wandb  # noqa: F401
    except ImportError as exc:
      raise SystemExit("wandb is not installed in this Python environment.") from exc

  manifest = [upload_run(args, row, args.dry_run) for row in rows]
  write_manifest(args, manifest)
  print(f"Wrote manifest: {args.output}")
  print(f"Runs: {len(manifest)}")


if __name__ == "__main__":
  main()
