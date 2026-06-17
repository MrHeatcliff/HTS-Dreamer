import csv
import json
import math
import os
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/mnt/disk1/backup_user/dat.tt2/xuance")
LOGROOT = ROOT / "logs/external_baselines/dreamerv3_official/full26_size12m"
OUT = ROOT / "external_baselines/dreamerv3-official/paper_artifacts/full26_log_reference_v16"
TARGET_GAMES = ["breakout", "alien"]
DISCOVERY_GAMES = ["asterix", "seaquest", "ms_pacman", "hero", "qbert", "crazy_climber"]


def dump_json(path, obj):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def write_csv(path, rows, fields=None):
  path.parent.mkdir(parents=True, exist_ok=True)
  if fields is None:
    fields = []
    for row in rows:
      for key in row:
        if key not in fields:
          fields.append(key)
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def ps_lines():
  cmd = "ps -eo pid,lstart,stat,cmd | grep 'dreamerv3.main' | grep -v grep || true"
  return subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE).stdout.splitlines()


def parse_proc(line):
  parts = line.split(None, 7)
  row = {"raw": line, "status": "running"}
  if len(parts) >= 8:
    row["pid"] = parts[0]
    row["start_time"] = " ".join(parts[1:6])
    row["ps_stat"] = parts[6]
    row["command"] = parts[7]
  else:
    row["pid"] = ""
    row["start_time"] = ""
    row["ps_stat"] = ""
    row["command"] = line
  cmd = row["command"]
  task = re.search(r"--task\s+(\S+)", cmd)
  seed = re.search(r"--seed\s+(\S+)", cmd)
  logdir = re.search(r"--logdir\s+(\S+)", cmd)
  row["task"] = task.group(1) if task else ""
  row["seed"] = seed.group(1) if seed else ""
  row["logdir"] = logdir.group(1) if logdir else ""
  return row


def process_observation():
  before = [parse_proc(x) for x in ps_lines()]
  return before


def extract_scalar(text, key):
  # Enough for the resolved DreamerV3 config files used here.
  m = re.search(rf"(?m)^\s*{re.escape(key)}:\s*([^\n#]+)", text)
  if not m:
    return "unknown"
  val = m.group(1).strip()
  return val


def recover_atari_repeat(text):
  task = extract_scalar(text, "task")
  # In resolved configs, atari100k env section contains repeat: 4 and use_seed.
  if isinstance(task, str) and task.startswith("atari100k"):
    return 4
  repeats = [int(x) for x in re.findall(r"(?m)^\s*repeat:\s*(\d+)", text)]
  return repeats[0] if repeats else "unknown"


def recover_protocol(logdir):
  cfgs = sorted(logdir.rglob("config.yaml"))
  text = cfgs[0].read_text(errors="replace") if cfgs else ""
  action_repeat = recover_atari_repeat(text) if text else "unknown"
  run_steps = extract_scalar(text, "steps") if text else "unknown"
  train_ratio = extract_scalar(text, "train_ratio") if text else "unknown"
  batch_size = extract_scalar(text, "batch_size") if text else "unknown"
  batch_length = extract_scalar(text, "batch_length") if text else "unknown"
  task = extract_scalar(text, "task") if text else "unknown"
  seed = extract_scalar(text, "seed") if text else "unknown"
  outputs = "unknown"
  m = re.search(r"(?m)^\s*outputs:\s*(.+)$", text)
  if m:
    outputs = m.group(1).strip()
  target_actions = "unknown"
  target_frames = "unknown"
  try:
    target_actions = float(run_steps)
    target_frames = target_actions * float(action_repeat)
  except Exception:
    pass
  return {
      "algorithm": "DreamerV3 official",
      "model_size": "size12m",
      "task": task,
      "seed": seed,
      "configs": "atari100k size12m",
      "logger_outputs": outputs,
      "action_repeat": action_repeat,
      "train_ratio": train_ratio,
      "batch_size": batch_size,
      "batch_length": batch_length,
      "replay_rate_or_updates_per_action": train_ratio,
      "eval_interval": "unknown",
      "checkpoint_interval": "unknown",
      "total_train_frames_or_actions_target": {
          "agent_actions": target_actions,
          "frames": target_frames,
      },
      "code_commit_if_available": "unknown",
      "config_file_paths": [str(x) for x in cfgs],
  }


def line_count(path):
  try:
    with path.open() as f:
      return sum(1 for _ in f)
  except Exception:
    return 0


def iter_jsonl(path):
  if not path.exists():
    return
  with path.open(errors="replace") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        yield json.loads(line)
      except Exception:
        yield {"_parse_error": line[:500]}


def read_jsonl(path):
  rows = []
  for row in iter_jsonl(path):
    rows.append(row)
  return rows


def scan_metric_summary(path):
  latest_step = ""
  has_episode = False
  for row in iter_jsonl(path):
    if "_parse_error" in row:
      continue
    step = row.get("step")
    if isinstance(step, (int, float)):
      latest_step = max(float(latest_step or 0), float(step))
    if "episode/score" in row:
      has_episode = True
  return latest_step, has_episode


def seed_from_dir(path):
  m = re.search(r"seed_(\d+)", path.name)
  return int(m.group(1)) if m else None


def game_seed_dirs():
  out = []
  if not LOGROOT.exists():
    return out
  for gdir in sorted([p for p in LOGROOT.iterdir() if p.is_dir()]):
    for sdir in sorted([p for p in gdir.iterdir() if p.is_dir() and p.name.startswith("seed_")]):
      out.append((gdir.name, seed_from_dir(sdir), sdir))
  return out


def latest_mtime(path):
  mt = 0
  candidates = [
      path,
      path / "config.yaml",
      path / "metrics.jsonl",
      path / "paper_artifacts/train_metrics.jsonl",
      path / "paper_artifacts/eval_metrics.jsonl",
      path / "paper_artifacts/episode_scores.jsonl",
      path / "paper_artifacts/latest_train_summary.json",
      path / "ckpt/latest",
  ]
  for p in candidates:
    try:
      mt = max(mt, p.stat().st_mtime)
    except OSError:
      pass
  return datetime.fromtimestamp(mt).isoformat() if mt else ""


def discover(processes):
  running_logdirs = {p.get("logdir", "") for p in processes if p.get("logdir")}
  rows = []
  for game, seed, logdir in game_seed_dirs():
    metric_candidates = [
        logdir / "metrics.jsonl",
        logdir / "scores.jsonl",
        logdir / "eval_metrics.jsonl",
        logdir / "train_metrics.jsonl",
        logdir / "paper_artifacts/train_metrics.jsonl",
        logdir / "paper_artifacts/eval_metrics.jsonl",
        logdir / "paper_artifacts/episode_scores.jsonl",
        logdir / "paper_artifacts/episode_scores.csv",
        logdir / "paper_artifacts/final_eval.json",
        logdir / "paper_artifacts/latest_train_summary.json",
    ]
    metric_files = [p for p in metric_candidates if p.exists()]
    jsonls = [p for p in metric_files if p.suffix == ".jsonl"]
    cfgs = [logdir / "config.yaml"] if (logdir / "config.yaml").exists() else []
    ckpt = [logdir / "ckpt/latest"] if (logdir / "ckpt/latest").exists() else []
    latest_step, has_episode = scan_metric_summary(logdir / "metrics.jsonl")
    proto = recover_protocol(logdir)
    target_frames = proto["total_train_frames_or_actions_target"].get("frames", "unknown")
    completed = "unknown"
    if latest_step != "" and isinstance(target_frames, (int, float)):
      completed = float(latest_step) >= target_frames * 0.99
    rows.append({
        "game": game,
        "seed": seed,
        "logdir": str(logdir),
        "exists": logdir.exists(),
        "process_running_now": str(logdir) in running_logdirs,
        "last_modified_time": latest_mtime(logdir),
        "config_files_present": bool(cfgs),
        "jsonl_logs_present": bool(jsonls),
        "scope_logs_present": (logdir / "scope").exists(),
        "wandb_local_logs_present": (logdir / "wandb").exists(),
        "checkpoint_present": bool(ckpt),
        "run_completed": completed,
        "training_frames_or_steps_available": latest_step != "",
        "eval_metrics_available": has_episode,
        "available_metric_files": [str(p) for p in metric_files],
        "latest_step": latest_step,
        "notes": "",
  })
  return rows


def compact_record(row):
  keys = ["step", "episode/score", "episode/length", "train/ret", "train/rew", "train/opt/updates", "replay/replay_ratio"]
  return {k: row[k] for k in keys if k in row}


def metric_rows_for(game, seed, logdir, discovery_row):
  proto = recover_protocol(logdir)
  action_repeat = proto["action_repeat"]
  target_frames = proto["total_train_frames_or_actions_target"].get("frames", "unknown")
  is_partial = True
  try:
    is_partial = float(discovery_row.get("latest_step") or 0) < float(target_frames) * 0.99
  except Exception:
    pass
  rows = []
  source_files = []
  if (logdir / "metrics.jsonl").exists():
    source_files.append((logdir / "metrics.jsonl", "metrics_jsonl"))
  if (logdir / "paper_artifacts/train_metrics.jsonl").exists():
    source_files.append((logdir / "paper_artifacts/train_metrics.jsonl", "paper_train_metrics_jsonl"))
  if (logdir / "paper_artifacts/episode_scores.jsonl").exists():
    source_files.append((logdir / "paper_artifacts/episode_scores.jsonl", "paper_episode_scores_jsonl"))
  for path, stype in source_files:
    try:
      fsize = path.stat().st_size
    except OSError:
      fsize = 0
    if stype == "paper_train_metrics_jsonl" and fsize > 100 * 1024 * 1024:
      rows.append({
          "game": game,
          "seed": seed,
          "logdir": str(logdir),
          "run_completed": discovery_row["run_completed"],
          "is_partial_run": is_partial,
          "metric_source_file": str(path),
          "metric_source_type": stype,
          "metric_name": "train_metrics_jsonl_skipped_large_file",
          "raw_step_key": "",
          "raw_step_value": "",
          "x_axis_name": "unknown",
          "x_axis_value": "",
          "agent_actions": "",
          "environment_steps": "",
          "frames": "",
          "action_repeat": action_repeat,
          "eval_return_mean": "",
          "eval_return_std": "",
          "eval_return_min": "",
          "eval_return_max": "",
          "eval_episodes": "",
          "train_return_mean_if_present": "",
          "timestamp": "",
          "wall_clock_time_if_present": "",
          "raw_json_excerpt_or_compact_record": json.dumps({"file_size_bytes": fsize}),
          "parse_status": "skipped",
          "parse_warning": "Skipped very large train_metrics.jsonl to avoid generating multi-GB derived artifacts; episode scores parsed from metrics.jsonl and episode_scores.jsonl.",
      })
      continue
    for rec in iter_jsonl(path):
      if "_parse_error" in rec:
        rows.append({
            "game": game, "seed": seed, "logdir": str(logdir), "run_completed": discovery_row["run_completed"],
            "is_partial_run": is_partial, "metric_source_file": str(path), "metric_source_type": stype,
            "metric_name": "parse_error", "parse_status": "fail", "parse_warning": rec["_parse_error"][:200],
        })
        continue
      step = rec.get("step", rec.get("env_step", rec.get("frame")))
      agent_actions = ""
      frames = ""
      warning = ""
      if isinstance(step, (int, float)):
        if isinstance(action_repeat, int):
          frames = float(step)
          agent_actions = float(step) / action_repeat
          warning = "x_axis recovered as logged step=frames; agent_actions=step/action_repeat"
        else:
          warning = "action_repeat missing; frames and agent_actions unknown"
      names = []
      if "episode/score" in rec:
        names.append("episode/score")
      if "train/ret" in rec:
        names.append("train/ret")
      if "train/rew" in rec:
        names.append("train/rew")
      if not names and any(k.startswith("train/") for k in rec):
        names.append("train_metrics_record")
      for name in names:
        rows.append({
            "game": game,
            "seed": seed,
            "logdir": str(logdir),
            "run_completed": discovery_row["run_completed"],
            "is_partial_run": is_partial,
            "metric_source_file": str(path),
            "metric_source_type": stype,
            "metric_name": name,
            "raw_step_key": "step" if "step" in rec else "unknown",
            "raw_step_value": step if step is not None else "",
            "x_axis_name": "frames" if frames != "" else "unknown",
            "x_axis_value": frames,
            "agent_actions": agent_actions,
            "environment_steps": frames,
            "frames": frames,
            "action_repeat": action_repeat,
            "eval_return_mean": rec.get("episode/score", "") if name == "episode/score" else "",
            "eval_return_std": "",
            "eval_return_min": rec.get("episode/score", "") if name == "episode/score" else "",
            "eval_return_max": rec.get("episode/score", "") if name == "episode/score" else "",
            "eval_episodes": 1 if name == "episode/score" else "",
            "train_return_mean_if_present": rec.get("train/ret", "") if name != "episode/score" else "",
            "timestamp": rec.get("timestamp", rec.get("time", "")),
            "wall_clock_time_if_present": rec.get("timer/duration", rec.get("wall_clock_time", "")),
            "raw_json_excerpt_or_compact_record": json.dumps(compact_record(rec), sort_keys=True),
            "parse_status": "pass",
            "parse_warning": warning + ("; episode/score is a single episode score, not an aggregated eval mean" if name == "episode/score" else ""),
        })
  return rows


def normalized_curve_rows(raw_rows):
  out = []
  for r in raw_rows:
    if r.get("metric_name") != "episode/score" or r.get("parse_status") != "pass":
      continue
    out.append({
        "game": r["game"],
        "seed": r["seed"],
        "partial_or_complete": "partial" if r["is_partial_run"] else "complete",
        "source": r["metric_source_type"],
        "source_file": r["metric_source_file"],
        "x_axis_agent_actions": r["agent_actions"],
        "x_axis_frames": r["frames"],
        "eval_return_mean": r["eval_return_mean"],
        "eval_return_std": r["eval_return_std"],
        "eval_episodes": r["eval_episodes"],
        "timestamp": r["timestamp"],
        "notes": r["parse_warning"],
    })
  return out


def latest_summary(discovery, curves):
  rows = []
  games = sorted(set([r["game"] for r in discovery if r["game"] in TARGET_GAMES + DISCOVERY_GAMES]))
  for game in games:
    seeds = sorted(set(int(r["seed"]) for r in discovery if r["game"] == game and str(r["seed"]).isdigit()))
    completed = []
    partial = []
    latest_actions = {}
    latest_frames = {}
    latest_returns = {}
    for seed in seeds:
      drow = next(r for r in discovery if r["game"] == game and int(r["seed"]) == seed)
      (completed if drow["run_completed"] is True else partial).append(seed)
      sub = [r for r in curves if r["game"] == game and int(r["seed"]) == seed and r["x_axis_frames"] != ""]
      if sub:
        last = max(sub, key=lambda x: float(x["x_axis_frames"]))
        latest_actions[str(seed)] = last["x_axis_agent_actions"]
        latest_frames[str(seed)] = last["x_axis_frames"]
        latest_returns[str(seed)] = last["eval_return_mean"]
    comp_vals = [float(latest_returns[str(s)]) for s in completed if str(s) in latest_returns]
    all_vals = [float(v) for v in latest_returns.values()]
    rows.append({
        "game": game,
        "available_seeds": json.dumps(seeds),
        "completed_seeds": json.dumps(completed),
        "partial_seeds": json.dumps(partial),
        "latest_agent_actions_per_seed": json.dumps(latest_actions, sort_keys=True),
        "latest_frames_per_seed": json.dumps(latest_frames, sort_keys=True),
        "latest_eval_return_mean_per_seed": json.dumps(latest_returns, sort_keys=True),
        "mean_latest_return_completed_only": statistics.mean(comp_vals) if comp_vals else "",
        "std_latest_return_completed_only": statistics.pstdev(comp_vals) if len(comp_vals) > 1 else (0.0 if comp_vals else ""),
        "mean_latest_return_all_available": statistics.mean(all_vals) if all_vals else "",
        "std_latest_return_all_available": statistics.pstdev(all_vals) if len(all_vals) > 1 else (0.0 if all_vals else ""),
        "notes_on_missing_or_partial_runs": "No completed run exists." if not completed else ("Partial seeds present: " + json.dumps(partial) if partial else "All available seeds completed."),
    })
  return rows


def plot_game(game, curves):
  rows = [r for r in curves if r["game"] == game and r["x_axis_frames"] != ""]
  if not rows:
    return False
  plt.figure(figsize=(6, 4))
  for seed in sorted(set(int(r["seed"]) for r in rows)):
    sub = sorted([r for r in rows if int(r["seed"]) == seed], key=lambda x: float(x["x_axis_frames"]))
    xs = [float(r["x_axis_frames"]) for r in sub]
    ys = [float(r["eval_return_mean"]) for r in sub]
    partial = any(r["partial_or_complete"] == "partial" for r in sub)
    plt.plot(xs, ys, label=f"seed {seed}" + (" partial" if partial else ""))
  plt.title(f"DreamerV3 official {game}")
  plt.xlabel("frames")
  plt.ylabel("episode/score")
  plt.legend(fontsize=7)
  plt.tight_layout()
  plt.savefig(OUT / f"fig_full26_{game}_learning_curve_v16.png", dpi=160)
  plt.close()
  return True


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  processes_before = process_observation()
  process_report = {"before": processes_before}
  discovery = discover(processes_before)
  present_games = {r["game"] for r in discovery}
  for game in TARGET_GAMES + DISCOVERY_GAMES:
    if game not in present_games:
      discovery.append({
          "game": game,
          "seed": "",
          "logdir": str(LOGROOT / game),
          "exists": False,
          "process_running_now": False,
          "last_modified_time": "",
          "config_files_present": False,
          "jsonl_logs_present": False,
          "scope_logs_present": False,
          "wandb_local_logs_present": False,
          "checkpoint_present": False,
          "run_completed": False,
          "training_frames_or_steps_available": False,
          "eval_metrics_available": False,
          "available_metric_files": [],
          "latest_step": "",
          "notes": "No log directory discovered under full26_size12m.",
      })
  wanted = [r for r in discovery if r["game"] in TARGET_GAMES + DISCOVERY_GAMES]

  raw_rows = []
  dmap = {(r["game"], int(r["seed"])): r for r in discovery if str(r["seed"]).isdigit()}
  for game in TARGET_GAMES:
    for g, seed, logdir in game_seed_dirs():
      if g == game:
        raw_rows.extend(metric_rows_for(g, seed, logdir, dmap[(g, int(seed))]))
  curves = normalized_curve_rows(raw_rows)
  summary = latest_summary(discovery, curves)
  protocol = []
  for game in TARGET_GAMES:
    for g, seed, logdir in game_seed_dirs():
      if g == game:
        row = recover_protocol(logdir)
        row["game"] = g
        row["logdir"] = str(logdir)
        protocol.append(row)
  plots = {game: plot_game(game, curves) for game in TARGET_GAMES}

  processes_after = process_observation()
  process_report["after"] = processes_after
  dump_json(OUT / "full26_process_observation_v16.json", process_report)
  lines = ["# Full26 Process Observation V16", "", "## Before", ""]
  for p in processes_before:
    lines.append(f"- pid `{p.get('pid')}` task `{p.get('task')}` seed `{p.get('seed')}` logdir `{p.get('logdir')}` status `{p.get('status')}`")
  lines += ["", "## After", ""]
  for p in processes_after:
    lines.append(f"- pid `{p.get('pid')}` task `{p.get('task')}` seed `{p.get('seed')}` logdir `{p.get('logdir')}` status `{p.get('status')}`")
  (OUT / "full26_process_observation_v16.md").write_text("\n".join(lines) + "\n")

  dump_json(OUT / "full26_log_discovery_v16.json", wanted)
  lines = ["# Full26 Log Discovery V16", ""]
  for r in wanted:
    lines.append(f"- {r['game']} seed {r['seed']}: completed=`{r['run_completed']}`, latest_step=`{r['latest_step']}`, metrics={len(r['available_metric_files'])}, running=`{r['process_running_now']}`")
  (OUT / "full26_log_discovery_v16.md").write_text("\n".join(lines) + "\n")

  write_csv(OUT / "full26_breakout_alien_metrics_v16.csv", raw_rows)
  dump_json(OUT / "full26_breakout_alien_metrics_v16.json", raw_rows)
  write_csv(OUT / "full26_learning_curves_normalized_v16.csv", curves)
  write_csv(OUT / "full26_latest_reference_summary_v16.csv", summary)
  lines = ["# Full26 Latest Reference Summary V16", ""]
  for r in summary:
    lines.append(f"- {r['game']}: completed={r['completed_seeds']} partial={r['partial_seeds']} latest_returns={r['latest_eval_return_mean_per_seed']} mean_completed={r['mean_latest_return_completed_only']} mean_all={r['mean_latest_return_all_available']}")
  (OUT / "full26_latest_reference_summary_v16.md").write_text("\n".join(lines) + "\n")
  dump_json(OUT / "full26_protocol_metadata_v16.json", protocol)
  lines = ["# Full26 Protocol Metadata V16", ""]
  for r in protocol:
    lines.append(f"- {r['game']} seed {r['seed']}: task `{r['task']}`, configs `{r['configs']}`, action_repeat `{r['action_repeat']}`, train_ratio `{r['train_ratio']}`, batch `{r['batch_size']}x{r['batch_length']}`, target `{r['total_train_frames_or_actions_target']}`")
  (OUT / "full26_protocol_metadata_v16.md").write_text("\n".join(lines) + "\n")

  interp = [
      "# Full26 Reference Interpretation V16",
      "",
      "## What the logs can support",
      "",
      "- External DreamerV3 reference for Breakout and Alien.",
      "- Rough learning-curve context from existing official log files.",
      "- Sanity check that official Atari logging is producing episode scores.",
      "- Future comparison target for HTS Atari Gate D2 after Gate D is unblocked.",
      "",
      "## What the logs cannot support",
      "",
      "- Synthetic mechanism claims.",
      "- HTS architecture selection.",
      "- Gate D1 pass/fail.",
      "- Gate D2 pass/fail.",
      "- HTS hyperparameter tuning.",
      "- Paper-final Atari benchmark unless run completion and protocol matching are verified.",
      "",
      "## Main caveats",
      "",
      "- `episode/score` is parsed as single episode score, not an aggregated evaluation mean.",
      "- The x-axis is recoverable as logged `step`; from config, `action_repeat=4`, so this report records `step` as frames and computes `agent_actions=step/4`.",
      "- Completed status is inferred by comparing latest logged step with `run.steps * action_repeat`; partial runs remain labeled.",
      "- Eval episode count for `episode/score` rows is recorded as `1`; aggregate eval episodes are not recovered from these logs.",
      "- These logs are not used to pass Gate D2 or tune HTS.",
      f"- Plot generation: {plots}.",
  ]
  (OUT / "full26_reference_interpretation_v16.md").write_text("\n".join(interp) + "\n")

  print(json.dumps({
      "output_dir": str(OUT),
      "raw_metric_rows": len(raw_rows),
      "curve_rows": len(curves),
      "summary_rows": len(summary),
      "plots": plots,
      "processes_before": processes_before,
      "processes_after": processes_after,
  }, indent=2))


if __name__ == "__main__":
  main()
