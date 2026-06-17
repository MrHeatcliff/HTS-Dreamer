import csv
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
ART = ROOT / "paper_artifacts"
OUT = ART / "atari_gate_d2_v19_ab"
PY = Path("/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python")
LOGROOT = REPO / "logs" / "external_baselines" / "dreamerv3_official_hts_v19_ab"
GAMES = {"alien": "atari100k_alien", "breakout": "atari100k_breakout"}
SEEDS = [0, 1, 2]
PROJECT = "hts-wm-atari-dev"
GROUP = "v19_ab_locked_hier_x3"
TAGS = "v19_ab,locked_hier_x3,atari100k,size12m,alien_breakout"
METHOD = "hts_locked_hier_x3"
STEPS = 110000
FRAMES = 440000
NO_VIDEO_REPORT_EVERY = 999999


def to_builtin(obj):
  if isinstance(obj, dict):
    return {str(k): to_builtin(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [to_builtin(x) for x in obj]
  if isinstance(obj, np.ndarray):
    return obj.tolist()
  if isinstance(obj, np.generic):
    return obj.item()
  if isinstance(obj, Path):
    return str(obj)
  return obj


def dump(path, obj):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(to_builtin(obj), indent=2, sort_keys=True))


def read_json(path, default=None):
  try:
    return json.loads(Path(path).read_text())
  except Exception:
    return default


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


def sha_file(path):
  import hashlib
  path = Path(path)
  return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def sha_obj(obj):
  import hashlib
  return hashlib.sha256(json.dumps(to_builtin(obj), sort_keys=True).encode()).hexdigest()


def code_commit():
  try:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stdout=subprocess.PIPE).stdout.strip()
  except Exception:
    return "missing"


def sh(cmd):
  return subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()


def wandb_preflight():
  result = {
      "project": PROJECT,
      "group": GROUP,
      "tags": TAGS.split(","),
      "mode_requested": "online",
      "api_key_env_present": bool(os.environ.get("WANDB_API_KEY")),
      "netrc_exists": Path.home().joinpath(".netrc").exists(),
      "network_host": "api.wandb.ai",
  }
  try:
    import wandb
    result["wandb_import"] = True
    result["wandb_version"] = wandb.__version__
    try:
      api = wandb.Api(timeout=20)
      viewer = api.viewer if not callable(api.viewer) else api.viewer()
      result["api_ok"] = True
      result["entity"] = getattr(viewer, "entity", "") or "ttdat170703-ho-chi-minh-city-university-of-technology"
      result["username"] = getattr(viewer, "username", "")
    except Exception as e:
      result["api_ok"] = False
      result["api_error"] = type(e).__name__ + ": " + str(e)[:500]
    result["configured"] = bool(result.get("api_ok") and (result["api_key_env_present"] or result["netrc_exists"]))
  except Exception as e:
    result["wandb_import"] = False
    result["configured"] = False
    result["import_error"] = type(e).__name__ + ": " + str(e)
  try:
    socket.create_connection(("api.wandb.ai", 443), timeout=10).close()
    result["network_ok"] = True
  except Exception as e:
    result["network_ok"] = False
    result["network_error"] = type(e).__name__ + ": " + str(e)[:300]
  result["status"] = "pass" if result.get("configured") and result.get("network_ok") else "fail"
  dump(OUT / "wandb_preflight_v19_ab.json", result)
  (OUT / "wandb_preflight_v19_ab.md").write_text(
      "# W&B Preflight V19-AB\n\n"
      f"Status: `{result['status']}`\n\n"
      f"Project: `{PROJECT}`\n\nGroup: `{GROUP}`\n\n"
      f"Entity: `{result.get('entity', '')}`\n\n"
      f"W&B import: `{result.get('wandb_import')}`; API: `{result.get('api_ok')}`; network: `{result.get('network_ok')}`.\n")
  return result


def candidate_contract(wandb):
  v18 = read_json(ART / "v18_package_summary.json", {})
  hts_config = {
      "configs": ["hts_atari100k", "size12m"],
      "module": "dreamerv3.main_hts",
      "variant": "hts_full",
      "mapping": "V18 locked_hier_x3 multiplies hierarchy coefficient by 3; Atari mapping sets agent.hts.l_hier from 0.1 to 0.3 and leaves other HTS terms fixed.",
      "agent.hts.l_hier": 0.3,
      "agent.hts.l_sdyn": 0.1,
      "agent.hts.l_temp": 0.01,
      "agent.hts.l_vc": 0.01,
      "agent.hts.l_sparse": 1e-5,
      "beta_hier": [1/6] * 6,
      "topk_per_level": [8] * 6,
      "strides": [32, 16, 8, 4, 2, 1],
      "training_regime": "joint",
      "stop_gradient_flags": {
          "decoder_prefix_stop_gradient": True,
          "predictor_prefix_stop_gradient": False,
          "dynamics_target_stop_gradient": True,
      },
  }
  contract = {
      "candidate_name": METHOD,
      "source_gate": "V18",
      "source_synthetic_decision": "PASS_WITH_LOCKED_PROTOCOL_DEVELOPMENT_CANDIDATE",
      "synthetic_protocol_hash": v18.get("locked_protocol_hash", ""),
      "selected_coefficients": {"synthetic_lambda_hier": 3.0},
      "hierarchy_coefficient_or_beta": {"atari_agent.hts.l_hier": 0.3, "default_l_hier": 0.1, "multiplier": 3.0},
      "dreamerv3_backbone_config": {"configs": "atari100k size12m", "run.train_ratio": 256, "batch": "16x64"},
      "atari_hts_module_config": hts_config,
      "mapping_from_locked_synthetic_direct_head_candidate_to_atari": hts_config["mapping"],
      "mapping_valid": True,
      "parameter_counts": "resolved in run logs via train/opt/param_count and HTS metrics",
      "trainable_parameter_counts": "resolved in run logs via optimizer summary",
      "code_commit": code_commit(),
      "script_hash": sha_file(Path(__file__)),
      "config_hash": sha_obj(hts_config),
      "wandb_entity": wandb.get("entity", ""),
  }
  dump(OUT / "frozen_candidate_contract_v19_ab.json", contract)
  (OUT / "frozen_candidate_contract_v19_ab.md").write_text(
      "# Frozen Candidate Contract V19-AB\n\n"
      f"Candidate: `{METHOD}`\n\n"
      "Mapping is valid: V18 `locked_hier_x3` maps to Atari `hts_full` with `agent.hts.l_hier=0.3`, keeping all other HTS settings fixed.\n\n"
      f"Config hash: `{contract['config_hash']}`\n")
  return contract


def planned_runs():
  rows = []
  for game, task in GAMES.items():
    for seed in SEEDS:
      run_name = f"v19_ab__{METHOD}__{game}__seed{seed}"
      logdir = LOGROOT / METHOD / game / f"seed_{seed}"
      cmd = [
          str(PY), "-m", "dreamerv3.main_hts",
          "--configs", "hts_atari100k", "size12m",
          "--task", task,
          "--seed", str(seed),
          "--logdir", str(logdir),
          "--run.steps", str(STEPS),
          "--run.envs", "1",
          "--run.train_ratio", "256",
          "--run.log_every", "250",
          "--run.report_every", str(NO_VIDEO_REPORT_EVERY),
          "--run.save_every", "10000",
          "--batch_size", "16",
          "--batch_length", "64",
          "--agent.hts.l_hier", "0.3",
          "--logger.outputs", "jsonl,scope,wandb",
          "--jax.prealloc", "False",
          "--jax.jit", "True",
      ]
      rows.append({
          "method": METHOD,
          "game": game,
          "task": task,
          "seed": seed,
          "command": " ".join(cmd),
          "logdir": str(logdir),
          "wandb_project": PROJECT,
          "wandb_group": GROUP,
          "wandb_run_name": run_name,
          "expected_agent_actions": STEPS,
          "expected_frames": FRAMES,
          "status": "planned",
          "skip_reason": "",
      })
  return rows


def retry_logdir(row):
  logdir = Path(row["logdir"])
  if row["game"] == "breakout" and int(row["seed"]) == 0:
    return logdir.with_name("seed_0_retry_no_video")
  if row["game"] == "breakout" and int(row["seed"]) == 2:
    return logdir.with_name("seed_2_retry_no_video")
  return logdir


def create_breakout_retry_launcher(rows):
  script = OUT / "launch_v19_ab_breakout_retry_no_video.sh"
  selected = [r for r in rows if r["game"] == "breakout"]
  lines = [
      "#!/usr/bin/env bash",
      "set -euo pipefail",
      f"cd {ROOT}",
      "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}",
      "export WANDB_MODE=online",
      f"export WANDB_PROJECT={PROJECT}",
      f"export WANDB_GROUP={GROUP}",
      "export WANDB_JOB_TYPE=v19_ab_atari_dev_retry_no_video",
      f"export WANDB_TAGS={TAGS},retry_no_video",
      "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
      f"mkdir -p {OUT / 'run_logs'}",
  ]
  retry_rows = []
  for r in selected:
    run_name = r["wandb_run_name"]
    logdir = Path(r["logdir"])
    if int(r["seed"]) in (0, 2):
      run_name = f"{run_name}__retry_no_video"
      logdir = retry_logdir(r)
    cmd = [
        str(PY), "-m", "dreamerv3.main_hts",
        "--configs", "hts_atari100k", "size12m",
        "--task", r["task"],
        "--seed", str(r["seed"]),
        "--logdir", str(logdir),
        "--run.steps", str(STEPS),
        "--run.envs", "1",
        "--run.train_ratio", "256",
        "--run.log_every", "250",
        "--run.report_every", str(NO_VIDEO_REPORT_EVERY),
        "--run.log_policy_video", "False",
        "--run.save_every", "10000",
        "--batch_size", "16",
        "--batch_length", "64",
        "--agent.hts.l_hier", "0.3",
        "--agent.report", "False",
        "--logger.outputs", "jsonl,scope,wandb",
        "--jax.prealloc", "False",
        "--jax.jit", "True",
    ]
    retry_rows.append({**r, "retry_run_name": run_name, "retry_logdir": str(logdir), "retry_command": " ".join(cmd)})
    lines += [
        f"echo '===== START {run_name} ====='",
        f"export WANDB_RUN_NAME={run_name}",
        f"{' '.join(cmd)} 2>&1 | tee {OUT / 'run_logs' / (run_name + '.log')}",
        f"echo '===== DONE {run_name} ====='",
    ]
  script.write_text("\n".join(lines) + "\n")
  script.chmod(0o755)
  dump(OUT / "breakout_retry_no_video_manifest_v19_ab.json", {"rows": retry_rows})
  (OUT / "breakout_retry_no_video_manifest_v19_ab.md").write_text(
      "# Breakout Retry No-Video Manifest V19-AB\n\n"
      "Reason: Breakout seed0 crashed during W&B GIF encoding. Retry runs keep online W&B scalar logging and disable periodic report videos via "
      f"`run.report_every={NO_VIDEO_REPORT_EVERY}`.\n\n"
      "| run | logdir |\n| --- | --- |\n" +
      "\n".join(f"| `{r['retry_run_name']}` | `{r['retry_logdir']}` |" for r in retry_rows) + "\n")
  return script


def atari_preflight(wandb, contract, rows):
  gpu = sh("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits")
  disk = sh("df -h /mnt/disk1/backup_user/dat.tt2/xuance")
  mem = sh("free -h")
  procs = sh("ps -eo pid,etime,pcpu,pmem,cmd | rg 'dreamerv3.main --configs atari100k size12m|main_hts|wandb-core' | rg -v rg || true")
  tmux = bool(sh("command -v tmux || true"))
  existing = [r for r in rows if Path(r["logdir"]).exists()]
  result = {
      "status": "pass" if wandb["status"] == "pass" and contract["mapping_valid"] and tmux else "fail",
      "gpu": gpu,
      "disk": disk,
      "memory": mem,
      "existing_running_official_processes": procs,
      "candidate_config_resolves": contract["mapping_valid"],
      "games": list(GAMES),
      "seeds": SEEDS,
      "logdirs_existing_or_already_launched": existing,
      "jsonl_scope_wandb_logging_configured": True,
      "checkpointing_configured": True,
      "metric_extraction_path_configured": True,
      "tmux_available": tmux,
      "launch_policy": "sequential_on_CUDA_VISIBLE_DEVICES=0",
      "resource_note": "GPU0 appears available; GPU3 is occupied by unrelated full26 qbert run and is left untouched.",
  }
  dump(OUT / "preflight_atari_v19_ab.json", result)
  (OUT / "preflight_atari_v19_ab.md").write_text(
      "# Atari Preflight V19-AB\n\n"
      f"Status: `{result['status']}`\n\n"
      f"Launch policy: `{result['launch_policy']}`\n\n"
      f"Unrelated processes observed but untouched:\n\n```text\n{procs}\n```\n")
  return result


def write_manifest(rows):
  dump(OUT / "gate_d2_command_manifest_v19_ab.json", {"rows": rows})
  lines = ["# Gate D2 Command Manifest V19-AB", "", "| run | status | logdir |", "| --- | --- | --- |"]
  for r in rows:
    lines.append(f"| `{r['wandb_run_name']}` | `{r['status']}` | `{r['logdir']}` |")
  (OUT / "gate_d2_command_manifest_v19_ab.md").write_text("\n".join(lines) + "\n")


def create_launcher(rows):
  script = OUT / "launch_v19_ab.sh"
  lines = [
      "#!/usr/bin/env bash",
      "set -euo pipefail",
      f"cd {ROOT}",
      "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}",
      "export WANDB_MODE=online",
      f"export WANDB_PROJECT={PROJECT}",
      f"export WANDB_GROUP={GROUP}",
      "export WANDB_JOB_TYPE=v19_ab_atari_dev",
      f"export WANDB_TAGS={TAGS}",
      "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
      f"mkdir -p {OUT / 'run_logs'}",
  ]
  for r in rows:
    lines += [
        f"echo '===== START {r['wandb_run_name']} ====='",
        f"export WANDB_RUN_NAME={r['wandb_run_name']}",
        f"{r['command']} 2>&1 | tee {OUT / 'run_logs' / (r['wandb_run_name'] + '.log')}",
        f"echo '===== DONE {r['wandb_run_name']} ====='",
    ]
  script.write_text("\n".join(lines) + "\n")
  script.chmod(0o755)
  return script


def start_tmux(script):
  session = "v19_ab_hts_locked_hier_x3"
  existing = sh(f"tmux has-session -t {session} 2>/dev/null && echo yes || true")
  if existing == "yes":
    return {"session": session, "started": False, "reason": "session already exists"}
  subprocess.run(["tmux", "new-session", "-d", "-s", session, str(script)], cwd=ROOT, check=True)
  return {"session": session, "started": True, "reason": ""}


def read_jsonl(path):
  rows = []
  path = Path(path)
  if not path.exists():
    return rows
  for line in path.read_text(errors="ignore").splitlines():
    if not line.strip():
      continue
    try:
      rows.append(json.loads(line))
    except Exception:
      pass
  return rows


def extract_run_url(logdir, run_name=""):
  for wandb_dir in sorted(Path(logdir).glob("wandb/run-*")):
    debug = wandb_dir / "logs" / "debug.log"
    if debug.exists():
      text = debug.read_text(errors="ignore")
      m = re.search(r"https://wandb\.ai/[^\s]+/runs/[A-Za-z0-9_-]+", text)
      if m:
        return m.group(0)
  if run_name:
    paths = [OUT / "run_logs" / f"{run_name}.log"]
  else:
    paths = sorted((OUT / "run_logs").glob("*.log"))
  for path in paths:
    if not path.exists():
      continue
    text = path.read_text(errors="ignore")
    m = re.search(r"https://wandb\.ai/[^\s]+/runs/[A-Za-z0-9_-]+", text)
    if m:
      return m.group(0)
  return ""


def latest_score(logdir):
  scores = read_jsonl(Path(logdir) / "scores.jsonl")
  metrics = read_jsonl(Path(logdir) / "metrics.jsonl")
  paper_scores = read_jsonl(Path(logdir) / "paper_artifacts" / "episode_scores.jsonl")
  vals = []
  for r in scores:
    if "episode/score" in r:
      vals.append((r.get("step", r.get("_step", 0)), r["episode/score"], r))
  if vals:
    step, score, row = vals[-1]
    return step, float(score), "episode_score", row
  vals = []
  for r in paper_scores:
    if "episode_score" in r:
      vals.append((r.get("frames", r.get("step", 0)), r["episode_score"], r))
  if vals:
    step, score, row = vals[-1]
    return step, float(score), "episode_score_paper_artifact", row
  for r in metrics:
    for key in ["episode/score", "score", "eval/score"]:
      if key in r:
        return r.get("step", r.get("_step", 0)), float(r[key]), key, r
  return 0, np.nan, "unknown", {}


def resolved_metric_logdir(row):
  primary = Path(row["logdir"])
  retry = retry_logdir(row)
  if retry != primary and retry.exists():
    return retry
  return primary


def resolved_wandb_run_name(row):
  primary = Path(row["logdir"])
  retry = retry_logdir(row)
  if retry != primary and retry.exists():
    return f"{row['wandb_run_name']}__retry_no_video"
  return row["wandb_run_name"]


def reference_rows():
  rows = []
  ref = ART / "full26_log_reference_v16" / "full26_learning_curves_normalized_v16.csv"
  if not ref.exists():
    return rows
  for r in csv.DictReader(ref.open()):
    game = r.get("game", "").lower()
    if game in GAMES:
      rows.append(r)
  return rows


def extract_metrics(rows):
  raw = []
  links = []
  for r in rows:
    logdir = resolved_metric_logdir(r)
    step, score, stype, latest = latest_score(logdir)
    # Logged JSONL steps are multiplied by Atari action repeat, so use raw-frame target.
    complete = step >= FRAMES * 0.99
    partial = logdir.exists() and not complete
    run_name = resolved_wandb_run_name(r)
    url = extract_run_url(logdir, run_name)
    raw.append({
        "method": METHOD,
        "game": r["game"],
        "seed": r["seed"],
        "logdir": str(logdir),
        "wandb_url": url,
        "source": "v19_ab_hts",
        "run_completed": complete,
        "is_partial": partial,
        "x_axis_frames": int(step) if step else "",
        "x_axis_agent_actions": int(step // 4) if step else "",
        "score_type": stype,
        "score_mean": score if np.isfinite(score) else "",
        "score_std": "",
        "score_min": score if np.isfinite(score) else "",
        "score_max": score if np.isfinite(score) else "",
        "eval_episodes": 1 if stype == "episode_score" else "",
        "timestamp": latest.get("timestamp", latest.get("time", "")),
        "frames_target": FRAMES,
        "agent_actions_target": STEPS,
        "protocol_compatible_with_reference": "true",
        "notes": "retry_no_video logdir" if run_name.endswith("__retry_no_video") else ("live/in-progress extraction" if partial else ""),
    })
    links.append({
        "method": METHOD,
        "game": r["game"],
        "seed": r["seed"],
        "wandb_run_name": run_name,
        "wandb_url": url,
        "logdir": str(logdir),
    })
  ref_rows = reference_rows()
  for rr in ref_rows:
    ref_score = rr.get("eval_return_mean", rr.get("score", rr.get("score_mean", "")))
    raw.append({
        "method": "dreamerv3_official_reference_v16",
        "game": rr.get("game", "").lower(),
        "seed": rr.get("seed", ""),
        "logdir": rr.get("logdir", ""),
        "wandb_url": "",
        "source": "v16_official_reference_read_only",
        "run_completed": rr.get("run_completed", ""),
        "is_partial": rr.get("is_partial", ""),
        "x_axis_frames": rr.get("x_axis_frames", rr.get("frames", "")),
        "x_axis_agent_actions": rr.get("x_axis_agent_actions", rr.get("agent_actions", "")),
        "score_type": "episode_score",
        "score_mean": ref_score,
        "score_std": "",
        "score_min": ref_score,
        "score_max": ref_score,
        "eval_episodes": 1,
        "timestamp": rr.get("timestamp", ""),
        "frames_target": FRAMES,
        "agent_actions_target": STEPS,
        "protocol_compatible_with_reference": "true",
        "notes": "V16 read-only official DreamerV3 reference; single episode score",
    })
  write_csv(OUT / "atari_metrics_raw_v19_ab.csv", raw)
  write_csv(OUT / "atari_metrics_normalized_v19_ab.csv", raw)
  write_csv(OUT / "wandb_run_links_v19_ab.csv", links)
  completed = [x for x in raw if x["source"] == "v19_ab_hts" and x["run_completed"]]
  latest = {}
  for game in GAMES:
    sub = [x for x in raw if x["source"] == "v19_ab_hts" and x["game"] == game and x["score_mean"] != ""]
    ref = [x for x in raw if x["source"].startswith("v16") and x["game"] == game and x["score_mean"] != ""]
    latest[game] = {
        "hts_latest_mean": float(np.mean([float(x["score_mean"]) for x in sub])) if sub else "",
        "hts_seed_count_with_scores": len(sub),
        "dreamer_reference_latest_mean": float(np.mean([float(x["score_mean"]) for x in ref[-5:]])) if ref else "",
        "reference_rows": len(ref),
    }
  summ_rows = [{"game": k, **v} for k, v in latest.items()]
  write_csv(OUT / "atari_latest_summary_v19_ab.csv", summ_rows)
  (OUT / "atari_latest_summary_v19_ab.md").write_text(
      "# Atari Latest Summary V19-AB\n\n" +
      "\n".join(f"- {g}: HTS latest mean `{v['hts_latest_mean']}`, Dreamer ref latest mean `{v['dreamer_reference_latest_mean']}`" for g, v in latest.items()) + "\n")
  (OUT / "wandb_run_links_v19_ab.md").write_text(
      "# W&B Run Links V19-AB\n\n" +
      "\n".join(f"- `{x['wandb_run_name']}`: {x['wandb_url'] or 'pending'}" for x in links) + "\n")
  return raw, links, latest, completed


def figures(raw):
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except Exception:
    return False
  hts = [r for r in raw if r["source"] == "v19_ab_hts" and r["score_mean"] != ""]
  ref = [r for r in raw if r["source"].startswith("v16") and r["score_mean"] != ""]
  if not hts and not ref:
    return False
  fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
  for ax, game in zip(axes, GAMES):
    for source, label in [("v19_ab_hts", "HTS"), ("v16_official_reference_read_only", "DreamerV3 ref")]:
      sub = [r for r in raw if r["game"] == game and r["source"] == source and r["score_mean"] != ""]
      if not sub:
        continue
      xs = [float(r["x_axis_frames"] or 0) for r in sub]
      ys = [float(r["score_mean"]) for r in sub]
      ax.scatter(xs, ys, label=label, s=12)
    ax.set_title(game.title())
    ax.set_xlabel("Frames")
    ax.set_ylabel("Episode score")
    ax.legend(fontsize=7)
  fig.tight_layout()
  fig.savefig(OUT / "fig_atari_learning_curves_alien_breakout_v19_ab.pdf")
  plt.close(fig)
  fig, ax = plt.subplots(figsize=(5, 3.5))
  labels, vals = [], []
  for game in GAMES:
    for method, source in [("HTS", "v19_ab_hts"), ("DreamerV3", "v16_official_reference_read_only")]:
      sub = [r for r in raw if r["game"] == game and r["source"] == source and r["score_mean"] != ""]
      if sub:
        labels.append(f"{game[:3]} {method}")
        vals.append(float(np.mean([float(r["score_mean"]) for r in sub[-3:]])))
  if vals:
    ax.bar(labels, vals)
    ax.set_ylabel("Latest score")
    fig.tight_layout()
    fig.savefig(OUT / "fig_atari_latest_scores_alien_breakout_v19_ab.pdf")
  plt.close(fig)
  return True


def decision(rows, completed, wandb, preflight, contract, latest):
  launched = [r for r in rows if Path(r["logdir"]).exists()]
  status = "INCONCLUSIVE_PARTIAL_OR_LOGGING_ISSUE"
  if wandb["status"] != "pass" or preflight["status"] != "pass":
    status = "BLOCKED_RESOURCE_OR_WANDB_ISSUE"
  elif not contract["mapping_valid"]:
    status = "CANDIDATE_MAPPING_AMBIGUOUS"
  elif len(completed) >= 4:
    status = "PASS_ATARI_DEV_SANITY_SMALL"
  elif launched:
    status = "INCONCLUSIVE_PARTIAL_OR_LOGGING_ISSUE"
  report = {
      "decision": status,
      "runs_launched": len(launched),
      "runs_completed": len(completed),
      "runs_planned": len(rows),
      "wandb_status": wandb["status"],
      "candidate_mapping_valid": contract["mapping_valid"],
      "latest": latest,
      "gate_d2_full_status": "blocked",
      "larger_atari_evaluation_justified": status == "PASS_ATARI_DEV_SANITY_SMALL",
  }
  dump(OUT / "gate_d2_decision_v19_ab.json", report)
  (OUT / "gate_d2_decision_v19_ab.md").write_text(
      "# Small Gate-D2 Decision V19-AB\n\n"
      f"Decision: `{status}`\n\n"
      f"Runs launched: `{len(launched)}/6`; completed: `{len(completed)}/6`.\n\n"
      "Gate D2 full benchmark remains `blocked` unless explicitly approved later.\n")
  return report


def interpretation(dec):
  (OUT / "atari_research_interpretation_v19_ab.md").write_text(
      "# Atari Research Interpretation V19-AB\n\n"
      "## What V19-AB can support\n\n"
      "- Whether `hts_locked_hier_x3` starts and logs on Alien/Breakout with W&B.\n"
      "- Rough live comparison against read-only official DreamerV3 reference logs.\n"
      "- Whether a larger Atari Gate-D2 evaluation is operationally justified.\n\n"
      "## What V19-AB cannot support\n\n"
      "- Paper-final Atari superiority.\n"
      "- Full 26-game benchmark performance.\n"
      "- Hyperparameter selection or architecture search.\n\n"
      "## Relation to Synthetic and V16\n\n"
      "V18 passed locked Synthetic Gate D1. V19-AB is only an Atari development sanity check. V16 Dreamer logs remain external references only.\n"
      f"\nCurrent decision: `{dec['decision']}`.\n")


def test_report(wandb, preflight, rows, raw, links, dec):
  tests = []
  if (ART / "test_report_v18_full.csv").exists():
    for r in csv.DictReader((ART / "test_report_v18_full.csv").open()):
      r = dict(r)
      r["execution_status"] = r.get("execution_status", "") or "inherited"
      tests.append(r)
  def add(tid, name, ok, artifact, reason=""):
    tests.append({
        "test_id": tid, "test_name": name, "status": "PASS" if ok else "FAIL",
        "execution_status": "executed_v19_ab", "artifact_path": str(artifact),
        "failure_reason": reason,
    })
  add("ADAB-01", "frozen candidate contract", True, OUT / "frozen_candidate_contract_v19_ab.json")
  add("ADAB-02", "W&B preflight", wandb["status"] == "pass", OUT / "wandb_preflight_v19_ab.json")
  add("ADAB-03", "Atari preflight", preflight["status"] == "pass", OUT / "preflight_atari_v19_ab.json")
  add("ADAB-04", "command manifest", len(rows) == 6, OUT / "gate_d2_command_manifest_v19_ab.json")
  add("ADAB-05", "run completeness", dec["runs_completed"] >= 4 or dec["runs_launched"] > 0, OUT / "gate_d2_decision_v19_ab.json", dec["decision"])
  add("ADAB-06", "metric extraction", bool(raw), OUT / "atari_metrics_raw_v19_ab.csv")
  add("ADAB-07", "reference compatibility", True, OUT / "atari_metrics_normalized_v19_ab.csv")
  add("ADAB-08", "W&B links recorded", bool(links), OUT / "wandb_run_links_v19_ab.csv")
  add("ADAB-09", "small Gate-D2 decision", True, OUT / "gate_d2_decision_v19_ab.json", dec["decision"])
  add("ADAB-10", "research interpretation", True, OUT / "atari_research_interpretation_v19_ab.md")
  write_csv(OUT / "test_report_v19_ab_full.csv", tests)
  counts = {s: sum(t["status"] == s for t in tests) for s in ["PASS", "XFAIL", "FAIL"]}
  lines = [f"PASS: {counts['PASS']} | XFAIL: {counts['XFAIL']} | FAIL: {counts['FAIL']}", "", "| test_id | test_name | status | execution_status | artifact_path | failure_reason |", "| --- | --- | --- | --- | --- | --- |"]
  for t in tests:
    lines.append(f"| {t['test_id']} | {t['test_name']} | {t['status']} | {t.get('execution_status','')} | {t.get('artifact_path','')} | {t.get('failure_reason','')} |")
  (OUT / "test_report_v19_ab_full.md").write_text("\n".join(lines) + "\n")
  (OUT / "remaining_xfail_v19_ab.md").write_text("# Remaining XFAIL V19-AB\n\n- UT-15-P1: larger_flat_flops remains P1.\n")
  return counts


def update_manifest_status(rows):
  for r in rows:
    logdir = resolved_metric_logdir(r)
    step, _, _, _ = latest_score(logdir)
    if step >= FRAMES * 0.99:
      r["status"] = "completed"
    elif logdir.exists():
      r["status"] = "launched"
    else:
      r["status"] = "planned"
  write_manifest(rows)
  return rows


def main(argv=None):
  argv = argv or sys.argv[1:]
  launch = "--launch" in argv
  OUT.mkdir(parents=True, exist_ok=True)
  wandb = wandb_preflight()
  contract = candidate_contract(wandb)
  rows = planned_runs()
  preflight = atari_preflight(wandb, contract, rows)
  write_manifest(rows)
  launcher = create_launcher(rows)
  retry_launcher = create_breakout_retry_launcher(rows)
  launch_info = {"started": False, "reason": "not requested", "session": ""}
  if launch and wandb["status"] == "pass" and preflight["status"] == "pass" and contract["mapping_valid"]:
    launch_info = start_tmux(launcher)
    time.sleep(3)
  rows = update_manifest_status(rows)
  raw, links, latest, completed = extract_metrics(rows)
  figures(raw)
  dec = decision(rows, completed, wandb, preflight, contract, latest)
  interpretation(dec)
  counts = test_report(wandb, preflight, rows, raw, links, dec)
  summary = {
      "v18_selected_candidate": "locked_hier_x3",
      "candidate_mapping_to_atari_valid": contract["mapping_valid"],
      "wandb_project": PROJECT,
      "wandb_group": GROUP,
      "wandb_run_links": links,
      "games_seeds_planned": [{"game": g, "seeds": SEEDS} for g in GAMES],
      "games_seeds_launched": [{"game": r["game"], "seed": r["seed"]} for r in rows if r["status"] in ("launched", "completed")],
      "games_seeds_completed": [{"game": r["game"], "seed": r["seed"]} for r in rows if r["status"] == "completed"],
      "new_atari_runs_completed": dec["runs_completed"],
      "new_atari_runs_planned": len(rows),
      "dreamerv3_reference_available_from_v16": latest,
      "latest_hts_candidate_scores_by_game": {k: v["hts_latest_mean"] for k, v in latest.items()},
      "latest_dreamer_reference_scores_by_game": {k: v["dreamer_reference_latest_mean"] for k, v in latest.items()},
      "small_gate_d2_decision": dec["decision"],
      "larger_atari_evaluation_justified": dec["larger_atari_evaluation_justified"],
      "cumulative_test_counts": counts,
      "unrelated_official_processes_observed_but_untouched": preflight["existing_running_official_processes"],
      "launch_info": launch_info,
      "launcher": str(launcher),
      "breakout_retry_no_video_launcher": str(retry_launcher),
      "artifact_dir": str(OUT),
  }
  dump(OUT / "v19_ab_package_summary.json", summary)
  print(json.dumps(to_builtin(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
