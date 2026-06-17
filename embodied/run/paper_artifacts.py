import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np


ATARI_HNS_REFERENCES = {
    "Alien": (227.8, 7127.7),
    "Amidar": (5.8, 1719.5),
    "Assault": (222.4, 742.0),
    "Asterix": (210.0, 8503.3),
    "BankHeist": (14.2, 753.1),
    "BattleZone": (2360.0, 37187.5),
    "Boxing": (0.1, 12.1),
    "Breakout": (1.7, 30.5),
    "ChopperCommand": (811.0, 7387.8),
    "CrazyClimber": (10780.5, 35829.4),
    "DemonAttack": (152.1, 1971.0),
    "Freeway": (0.0, 29.6),
    "Frostbite": (65.2, 4334.7),
    "Gopher": (257.6, 2412.5),
    "Hero": (1027.0, 30826.4),
    "Jamesbond": (29.0, 302.8),
    "Kangaroo": (52.0, 3035.0),
    "Krull": (1598.0, 2665.5),
    "KungFuMaster": (258.5, 22736.3),
    "MsPacman": (307.3, 6951.6),
    "Pong": (-20.7, 14.6),
    "PrivateEye": (24.9, 69571.3),
    "Qbert": (163.9, 13455.0),
    "RoadRunner": (11.5, 7845.0),
    "Seaquest": (68.4, 42054.7),
    "UpNDown": (533.4, 11693.2),
}


EPISODE_FIELDS = [
    "experiment_id",
    "suite",
    "task",
    "condition",
    "method",
    "seed",
    "step",
    "env_steps",
    "agent_actions",
    "frames",
    "action_repeat",
    "episode_index",
    "episode_score",
    "episode_length",
    "episode_hns",
    "optimizer_updates",
    "train_ratio_replayed_steps_per_agent_action",
    "batch_size",
    "batch_length",
    "minibatch_steps",
    "expected_updates_per_agent_action",
    "realized_optimizer_updates",
    "realized_agent_actions",
    "expected_updates_per_raw_frame",
    "realized_frames",
    "param_count",
    "fps_policy",
    "fps_train",
    "wall_clock_seconds",
    "config_hash",
    "code_commit",
    "wandb_project",
    "wandb_run_name",
]


def _to_builtin(value):
  if isinstance(value, dict):
    return {str(key): _to_builtin(val) for key, val in value.items()}
  if hasattr(value, "items") and not isinstance(value, (str, bytes)):
    try:
      return {str(key): _to_builtin(val) for key, val in value.items()}
    except Exception:
      pass
  if isinstance(value, (list, tuple)):
    return [_to_builtin(item) for item in value]
  if isinstance(value, np.ndarray):
    if value.ndim == 0:
      return _to_builtin(value.item())
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, Path):
    return str(value)
  return value


def _append_jsonl(path, row):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a") as file:
    file.write(json.dumps(_to_builtin(row), sort_keys=True) + "\n")


def _append_csv(path, fields, row):
  path.parent.mkdir(parents=True, exist_ok=True)
  exists = path.exists()
  with path.open("a", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
    if not exists:
      writer.writeheader()
    writer.writerow({key: _to_builtin(row.get(key, "")) for key in fields})


def _compact_metrics(metrics):
  compact = {}
  for key, value in metrics.items():
    if key == "timer" or key.startswith("timer/"):
      continue
    value = _to_builtin(value)
    if isinstance(value, (int, float, str, bool)) or value is None:
      compact[key] = value
    elif isinstance(value, list) and len(value) <= 16:
      compact[key] = value
  return compact


def _git_commit():
  try:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
  except Exception:
    return "unknown"


def _config_hash(config):
  payload = json.dumps(
      _to_builtin(config), sort_keys=True, default=str).encode("utf-8")
  return hashlib.sha256(payload).hexdigest()[:16]


def _atari_title(name):
  aliases = {
      "bank_heist": "BankHeist",
      "battle_zone": "BattleZone",
      "chopper_command": "ChopperCommand",
      "crazy_climber": "CrazyClimber",
      "demon_attack": "DemonAttack",
      "james_bond": "Jamesbond",
      "kung_fu_master": "KungFuMaster",
      "ms_pacman": "MsPacman",
      "private_eye": "PrivateEye",
      "road_runner": "RoadRunner",
      "up_n_down": "UpNDown",
  }
  if name in aliases:
    return aliases[name]
  return "".join(part.capitalize() for part in name.split("_"))


def _task_name(task):
  if task.startswith("atari100k_"):
    return _atari_title(task[len("atari100k_"):])
  if task.startswith("atari_"):
    return _atari_title(task[len("atari_"):])
  return task


def _suite(task):
  if task.startswith("atari100k_"):
    return "atari100k"
  return task.split("_")[0] if "_" in task else "unknown"


def _hns(task, score):
  if task not in ATARI_HNS_REFERENCES:
    return None
  random, human = ATARI_HNS_REFERENCES[task]
  denom = human - random
  return None if abs(denom) < 1e-12 else 100.0 * (score - random) / denom


class PaperArtifactWriter:

  def __init__(self, logdir, args):
    self.logdir = Path(str(logdir))
    self.root = self.logdir / "paper_artifacts"
    self.root.mkdir(parents=True, exist_ok=True)
    self.start_time = time.time()
    self.args = args
    self.task = _task_name(str(args.task))
    self.suite = _suite(str(args.task))
    self.method = os.environ.get("PAPER_METHOD", "DreamerV3")
    self.condition = os.environ.get("PAPER_CONDITION", "official")
    self.experiment_id = os.environ.get(
        "PAPER_EXPERIMENT_ID", "official_dreamerv3")
    self.action_repeat = self._action_repeat()
    self.latent_anchor = self._latent_anchor()
    self.config_hash = _config_hash(args)
    self.code_commit = _git_commit()
    self.last_train = {}
    self.first_update_step = None
    self.first_post_prefill_step = None
    self.episode_index = 0
    self._write_meta()

  def _action_repeat(self):
    envkey = str(self.args.task).split("_")[0]
    try:
      return int(self.args.env.get(envkey, {}).get("repeat", 1))
    except Exception:
      return 1

  def _latent_anchor(self):
    hts = {}
    try:
      hts = self.args.agent.get("hts", {})
    except Exception:
      pass
    try:
      rssm = self.args.agent.dyn.rssm
      dim = int(rssm.deter) + int(rssm.stoch) * int(rssm.classes)
    except Exception:
      dim = hts.get("latent_anchor_dim", "")
    return {
        "latent_anchor_name": hts.get("latent_anchor_name", "rssm_repfeat"),
        "latent_anchor_source_module": hts.get(
            "latent_anchor_source_module", "dreamerv3.rssm.RSSM.loss"),
        "latent_anchor_dim": dim,
    }

  def _base(self, step):
    step = int(step)
    updates = self.last_train.get("train/opt/updates", "")
    batch_size = int(getattr(self.args, "batch_size", 0))
    batch_length = int(getattr(self.args, "batch_length", 0))
    minibatch_steps = batch_size * batch_length
    train_ratio = float(getattr(self.args, "train_ratio", 0.0))
    expected_updates_per_agent_action = (
        train_ratio / minibatch_steps if minibatch_steps else 0.0)
    return {
        "experiment_id": self.experiment_id,
        "suite": self.suite,
        "task": self.task,
        "condition": self.condition,
        "method": self.method,
        "seed": int(self.args.seed),
        "step": step,
        "env_steps": step,
        "agent_actions": step,
        "frames": step * self.action_repeat,
        "action_repeat": self.action_repeat,
        "batch_size": batch_size,
        "batch_length": batch_length,
        "sequence_length": batch_length,
        "minibatch_steps": minibatch_steps,
        "latent_anchor_name": self.latent_anchor["latent_anchor_name"],
        "latent_anchor_source_module": self.latent_anchor[
            "latent_anchor_source_module"],
        "latent_anchor_dim": self.latent_anchor["latent_anchor_dim"],
        "optimizer_updates": updates,
        "train_ratio_replayed_steps_per_agent_action": train_ratio,
        "expected_updates_per_agent_action": expected_updates_per_agent_action,
        "realized_optimizer_updates": updates,
        "realized_agent_actions": step,
        "expected_updates_per_raw_frame": (
            expected_updates_per_agent_action / self.action_repeat
            if self.action_repeat else 0.0),
        "realized_frames": step * self.action_repeat,
        "param_count": self.last_train.get("train/opt/param_count", ""),
        "fps_policy": self.last_train.get("fps/policy", ""),
        "fps_train": self.last_train.get("fps/train", ""),
        "wall_clock_seconds": round(time.time() - self.start_time, 3),
        "config_hash": self.config_hash,
        "code_commit": self.code_commit,
        "wandb_project": os.environ.get("WANDB_PROJECT", ""),
        "wandb_run_name": os.environ.get("WANDB_RUN_NAME", ""),
    }

  def _write_meta(self):
    meta = {
        "experiment_id": self.experiment_id,
        "suite": self.suite,
        "task": self.task,
        "condition": self.condition,
        "method": self.method,
        "seed": int(self.args.seed),
        "logdir": str(self.logdir),
        "command": " ".join(os.sys.argv),
        "config_hash": self.config_hash,
        "code_commit": self.code_commit,
        "action_repeat": self.action_repeat,
        "config": _to_builtin(self.args),
        "env": _to_builtin(getattr(self.args, "env", {})),
        "run": _to_builtin(getattr(self.args, "run", {})),
        "batch_size": int(self.args.batch_size),
        "batch_length": int(self.args.batch_length),
        "minibatch_steps": int(self.args.batch_size) * int(self.args.batch_length),
        "train_ratio_replayed_steps_per_agent_action": float(self.args.train_ratio),
        "expected_updates_per_agent_action": (
            float(self.args.train_ratio) /
            (int(self.args.batch_size) * int(self.args.batch_length))),
        "expected_updates_per_raw_frame": (
            float(self.args.train_ratio) /
            (int(self.args.batch_size) * int(self.args.batch_length)) /
            self.action_repeat),
        "replay_semantics": {
            "configured_train_ratio_units": (
                "replayed environment timesteps per agent action"),
            "expected_update_rate_units": (
                "optimizer minibatch updates per agent action"),
            "initial_prefill_excluded_from_consistency_check": True,
            "compilation_steps_excluded_from_consistency_check": True,
        },
        "sequence_length": int(self.args.batch_length),
        **self.latent_anchor,
        "wandb": {
            "project": os.environ.get("WANDB_PROJECT", ""),
            "entity": os.environ.get("WANDB_ENTITY", ""),
            "mode": os.environ.get("WANDB_MODE", ""),
            "group": os.environ.get("WANDB_GROUP", ""),
            "job_type": os.environ.get("WANDB_JOB_TYPE", ""),
            "run_name": os.environ.get("WANDB_RUN_NAME", ""),
            "tags": os.environ.get("WANDB_TAGS", ""),
        },
        "created_wall_time": self.start_time,
    }
    with (self.root / "run_meta.json").open("w") as file:
      json.dump(_to_builtin(meta), file, indent=2, sort_keys=True)
    (self.root / "eval_metrics.jsonl").touch(exist_ok=True)
    final_eval = {
        "status": "not_run",
        "reason": "final evaluator has not been launched for this run",
        "eval_episodes": 0,
        **self._base(0),
    }
    with (self.root / "final_eval.json").open("w") as file:
      json.dump(_to_builtin(final_eval), file, indent=2, sort_keys=True)
    checkpoints = {
        "status": "initialized",
        "checkpoint_rule": "final_checkpoint_for_headline_tables",
        "logdir": str(self.logdir),
        "ckpt_dir": str(self.logdir / "ckpt"),
    }
    with (self.root / "checkpoints_manifest.json").open("w") as file:
      json.dump(_to_builtin(checkpoints), file, indent=2, sort_keys=True)
    self._write_replay_consistency(step=0, status="pending")

  def write_episode(self, step, score, length):
    self.episode_index += 1
    score = float(score)
    row = self._base(step)
    row.update({
        "episode_index": self.episode_index,
        "episode_score": score,
        "episode_length": float(length),
        "episode_hns": _hns(self.task, score),
    })
    _append_jsonl(self.root / "episode_scores.jsonl", row)
    _append_csv(self.root / "episode_scores.csv", EPISODE_FIELDS, row)

  def write_eval_episode(self, step, score, length):
    row = self._base(step)
    score = float(score)
    row.update({
        "episode_index": self.episode_index + 1,
        "episode_score": score,
        "episode_length": float(length),
        "episode_hns": _hns(self.task, score),
        "split": "eval",
    })
    self.episode_index += 1
    _append_jsonl(self.root / "eval_metrics.jsonl", row)

  def write_train_metrics(self, step, metrics):
    flat = _compact_metrics({str(k): v for k, v in metrics.items()})
    self.last_train.update(flat)
    try:
      updates = float(self.last_train.get("train/opt/updates", 0.0))
      if updates > 0 and self.first_update_step is None:
        self.first_update_step = int(step)
    except Exception:
      pass
    row = self._base(step)
    row.update(flat)
    _append_jsonl(self.root / "train_metrics.jsonl", row)
    summary = dict(row)
    with (self.root / "latest_train_summary.json").open("w") as file:
      json.dump(_to_builtin(summary), file, indent=2, sort_keys=True)
    self._write_replay_consistency(step)

  def write_update_event(
      self, step, requested_updates, executed_updates,
      optimizer_updates_cumulative, is_prefill=False, is_compile_only=False,
      scheduler_accumulator_before=None, scheduler_accumulator_after=None):
    row = self._base(step)
    minibatch_steps = int(row.get("minibatch_steps") or 0)
    if not is_prefill and self.first_post_prefill_step is None:
      self.first_post_prefill_step = int(step)
    post_prefill = (
        "" if self.first_post_prefill_step is None or is_prefill
        else int(step) - int(self.first_post_prefill_step) + 1)
    row.update({
        "agent_action_index": int(step),
        "post_prefill_agent_action_index": post_prefill,
        "is_prefill": bool(is_prefill),
        "is_compile_only": bool(is_compile_only),
        "ratio_scheduler_requested_updates": int(requested_updates),
        "optimizer_updates_executed": int(executed_updates),
        "optimizer_updates_cumulative": float(optimizer_updates_cumulative),
        "replayed_timesteps_cumulative": (
            float(optimizer_updates_cumulative) * minibatch_steps),
        "scheduler_accumulator_before": scheduler_accumulator_before,
        "scheduler_accumulator_after": scheduler_accumulator_after,
    })
    _append_jsonl(
        self.root / "replay_consistency_v4" / "update_event_trace.jsonl",
        row)
    _append_jsonl(
        self.root / "replay_consistency_v6" / "update_event_trace_v6.jsonl",
        row)

  def _write_replay_consistency(self, step, status=None):
    base = self._base(step)
    expected = float(base["expected_updates_per_agent_action"])
    try:
      updates = float(base["realized_optimizer_updates"])
    except Exception:
      updates = 0.0
    start = self.first_update_step
    if status is None:
      if updates <= 0 or start is None or int(step) <= start:
        status = "pending"
      else:
        status = "pass"
    denom = max(int(step) - int(start or 0), 0)
    realized = None if status == "pending" else (updates / denom if denom else None)
    tolerance = float(os.environ.get("PAPER_REPLAY_RATIO_TOLERANCE", "0.05"))
    abs_error = None if realized is None else abs(realized - expected)
    if status != "pending" and realized is not None and abs_error >= tolerance:
      status = "fail"
    row = {
        **base,
        "status": status,
        "first_update_step": start,
        "realized_agent_actions_excluding_prefill": denom,
        "realized_updates_per_agent_action_excluding_prefill": realized,
        "absolute_error": abs_error,
        "tolerance": tolerance,
        "initial_prefill_excluded": True,
        "compilation_steps_excluded": True,
    }
    with (self.root / "replay_ratio_consistency.json").open("w") as file:
      json.dump(_to_builtin(row), file, indent=2, sort_keys=True)

  def finalize(self, step, checkpoint_path=None):
    train_metrics = self.root / "train_metrics.jsonl"
    if not train_metrics.exists() or train_metrics.stat().st_size == 0:
      self.write_train_metrics(step, {
          "paper/native_final_train_metrics_flush": True,
          "paper/native_final_train_metrics_reason": (
              "train loop ended before a populated periodic logger row"),
      })
    self._write_replay_consistency(step)
    final_eval = {
        "status": "not_run",
        "reason": "final evaluator has not been launched for this run",
        "checkpoint_path": str(checkpoint_path or ""),
        "global_step": int(step),
        "eval_episodes": 0,
        "sequence_length": int(getattr(self.args, "batch_length", 0)),
        "peak_memory_mb": self.last_train.get("usage/gpu_mem", ""),
        "updates_per_second": self.last_train.get("fps/train", ""),
        "environment_steps_per_second": self.last_train.get("fps/policy", ""),
        **self._base(step),
    }
    with (self.root / "final_eval.json").open("w") as file:
      json.dump(_to_builtin(final_eval), file, indent=2, sort_keys=True)
    checkpoints = {
        "status": "training_finished",
        "checkpoint_rule": "latest_checkpoint_after_train_loop",
        "checkpoint_path": str(checkpoint_path or ""),
        "global_step": int(step),
        "logdir": str(self.logdir),
        "ckpt_dir": str(self.logdir / "ckpt"),
    }
    with (self.root / "checkpoints_manifest.json").open("w") as file:
      json.dump(_to_builtin(checkpoints), file, indent=2, sort_keys=True)

  def finalize_eval(self, step, checkpoint_path, eval_episodes, scores, lengths):
    scores = [float(x) for x in scores]
    lengths = [float(x) for x in lengths]
    final_eval = {
        "status": "complete",
        "checkpoint_path": str(checkpoint_path or ""),
        "global_step": int(step),
        "eval_episodes": int(eval_episodes),
        "eval_score_mean": float(np.mean(scores)) if scores else None,
        "eval_score_std": float(np.std(scores)) if scores else None,
        "eval_length_mean": float(np.mean(lengths)) if lengths else None,
        "sequence_length": int(getattr(self.args, "batch_length", 0)),
        "peak_memory_mb": self.last_train.get("usage/gpu_mem", ""),
        "updates_per_second": self.last_train.get("fps/train", ""),
        "environment_steps_per_second": self.last_train.get("fps/policy", ""),
        **self._base(step),
    }
    with (self.root / "final_eval.json").open("w") as file:
      json.dump(_to_builtin(final_eval), file, indent=2, sort_keys=True)
