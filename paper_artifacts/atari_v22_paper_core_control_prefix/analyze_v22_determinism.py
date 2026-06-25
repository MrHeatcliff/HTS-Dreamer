#!/usr/bin/env python3
import argparse
import json
import math
import struct
from pathlib import Path


SCALARS = [
    "episode-score.float",
    "train-loss-total.float",
    "train-loss-weighted-wm.float",
    "train-loss-weighted-hier.float",
    "train-loss-weighted-sdyn.float",
    "train-loss-weighted-ctrl.float",
    "train-hts-aux_warmup_alpha.float",
    "train-hts-active_ratio.float",
    "train-hts-z_full_variance.float",
    "replay-replay_ratio.float",
]


def read_float_file(path):
  if not path.exists():
    return []
  data = path.read_bytes()
  rows = []
  for idx in range(0, len(data), 16):
    if idx + 16 <= len(data):
      rows.append(struct.unpack(">Qd", data[idx:idx + 16]))
  return rows


def read_jsonl(path):
  rows = []
  if not path.exists():
    return rows
  for line in path.read_text().splitlines():
    if not line.strip():
      continue
    try:
      rows.append(json.loads(line))
    except json.JSONDecodeError:
      pass
  return rows


def first_mismatch(a, b, atol=0.0):
  n = min(len(a), len(b))
  for idx in range(n):
    if a[idx][0] != b[idx][0]:
      return idx, a[idx], b[idx], "step"
    av, bv = a[idx][1], b[idx][1]
    if not (math.isfinite(av) and math.isfinite(bv)):
      if av != bv:
        return idx, a[idx], b[idx], "finite"
    elif abs(av - bv) > atol:
      return idx, a[idx], b[idx], "value"
  if len(a) != len(b):
    return n, a[n] if n < len(a) else None, b[n] if n < len(b) else None, "length"
  return None


def first_jsonl_mismatch(a, b, keys):
  n = min(len(a), len(b))
  for idx in range(n):
    for key in keys:
      if a[idx].get(key) != b[idx].get(key):
        return {
            "index": idx,
            "key": key,
            "base": {k: a[idx].get(k) for k in keys if k in a[idx]},
            "other": {k: b[idx].get(k) for k in keys if k in b[idx]},
        }
  if len(a) != len(b):
    return {
        "index": n,
        "key": "length",
        "base_len": len(a),
        "other_len": len(b),
    }
  return None


def run_summary(path):
  scope = path / "scope"
  scores = read_float_file(scope / "episode-score.float")
  metrics_jsonl = read_jsonl(path / "metrics.jsonl")
  return {
      "path": str(path),
      "exists": path.exists(),
      "scope_exists": scope.exists(),
      "score_count": len(scores),
      "score_latest": scores[-1] if scores else None,
      "score_max": max([x[1] for x in scores], default=None),
      "score_mean_last20": (
          sum(x[1] for x in scores[-20:]) / min(20, len(scores))
          if scores else None),
      "metrics_jsonl_rows": len(metrics_jsonl),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", required=True)
  parser.add_argument("--repeats", default="0 1")
  parser.add_argument("--atol", type=float, default=0.0)
  parser.add_argument("--output", default="")
  args = parser.parse_args()

  root = Path(args.root)
  repeat_ids = args.repeats.split()
  paths = [root / f"repeat_{rid}" for rid in repeat_ids]
  report = {
      "root": str(root),
      "repeats": repeat_ids,
      "summaries": [run_summary(path) for path in paths],
      "comparisons": [],
  }

  if len(paths) >= 2:
    base = paths[0]
    for other in paths[1:]:
      comp = {"base": str(base), "other": str(other), "scalars": {}}
      for name in SCALARS:
        a = read_float_file(base / "scope" / name)
        b = read_float_file(other / "scope" / name)
        mismatch = first_mismatch(a, b, args.atol)
        comp["scalars"][name] = {
            "base_count": len(a),
            "other_count": len(b),
            "exact_match": mismatch is None,
            "first_mismatch": (
                None if mismatch is None else {
                    "index": mismatch[0],
                    "base": mismatch[1],
                    "other": mismatch[2],
                    "kind": mismatch[3],
                }),
        }
      action_keys = [
          "step", "worker", "optimizer_updates_cumulative",
          "transition_hash", "action_hash", "reward", "is_first",
          "is_last", "is_terminal", "actions"]
      batch_keys = [
          "step", "update_index_in_step", "optimizer_updates_before",
          "optimizer_updates_after", "batch_hash", "stepid_hash",
          "reward_hash", "action_hash", "is_first_hash",
          "is_last_hash", "is_terminal_hash", "train/opt/loss",
          "train/opt/updates", "train/opt/grad_norm",
          "train/opt/update_rms"]
      actions_a = read_jsonl(base / "paper_artifacts" / "determinism" / "action_trace.jsonl")
      actions_b = read_jsonl(other / "paper_artifacts" / "determinism" / "action_trace.jsonl")
      batches_a = read_jsonl(base / "paper_artifacts" / "determinism" / "batch_trace.jsonl")
      batches_b = read_jsonl(other / "paper_artifacts" / "determinism" / "batch_trace.jsonl")
      comp["action_trace"] = {
          "base_count": len(actions_a),
          "other_count": len(actions_b),
          "first_mismatch": first_jsonl_mismatch(
              actions_a, actions_b, action_keys),
      }
      comp["batch_trace"] = {
          "base_count": len(batches_a),
          "other_count": len(batches_b),
          "first_mismatch": first_jsonl_mismatch(
              batches_a, batches_b, batch_keys),
      }
      report["comparisons"].append(comp)

  text = json.dumps(report, indent=2)
  print(text)
  if args.output:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
  main()
