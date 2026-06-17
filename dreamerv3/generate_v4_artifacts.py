import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from . import component_matrix


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
PAPER = REPO_ROOT / "paper.txt"
OUT = ROOT / "paper_artifacts"
PAPER_HASH = "9c511de120655c2c6038ce46c4c613dab9e9c3ae83db01a4b86126745b0276a6"
CONFIG_NAMES = [
    "dreamer_anchor",
    "hts_full",
    "flat_sae",
    "flat_mh",
    "flat_partition_dim_matched",
    "sgf_style_flat_same_code",
    "recon_only_hierarchy",
    "matryoshka_only",
    "dense_multistride_no_sparse",
    "larger_flat_param",
    "larger_flat_flops",
    "hts_no_temp",
    "hts_no_vc",
    "hts_no_hier",
    "hts_no_sdyn",
]


def _write(path, text):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text)


def _dump(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=2, sort_keys=True))


def _paper_text():
  return PAPER.read_text()


def _paper_hash():
  return hashlib.sha256(PAPER.read_bytes()).hexdigest()


def _find_headings(text, prefix):
  pattern = re.compile(r"\\(?:sub)*section\{([^}]+)\}")
  rows = []
  for match in pattern.finditer(text):
    start = text[:match.start()].count("\n") + 1
    title = match.group(1)
    if prefix(title):
      rows.append({"line": start, "heading": title})
  return rows


def _find_labels(text, kind):
  labels = []
  for match in re.finditer(r"\\label\{" + re.escape(kind) + r":([^}]+)\}", text):
    labels.append({
        "line": text[:match.start()].count("\n") + 1,
        "label": f"{kind}:{match.group(1)}",
    })
  return labels


def _context_lines(text, needle, before=2, after=3):
  lines = text.splitlines()
  matches = []
  for idx, line in enumerate(lines):
    if needle.lower() in line.lower():
      lo = max(0, idx - before)
      hi = min(len(lines), idx + after + 1)
      matches.append({
          "line": idx + 1,
          "excerpt": "\n".join(f"{i + 1}: {lines[i]}" for i in range(lo, hi)),
      })
  return matches


def contract_note():
  text = _paper_text()
  sha = _paper_hash()
  method_headings = _find_headings(
      text, lambda title: title in (
          "Hierarchical Sparse Latents",
          "Nested Reconstruction",
          "Multi-Stride Sparse Dynamics",
          "Temporal Consistency and Anti-Collapse",
          "Variance--Covariance Regularization",
          "Sparsity",
          "Full Objective and Training Regimes"))
  experiment_headings = _find_headings(
      text, lambda title: any(key in title for key in (
          "Experimental", "Benchmark", "Atari", "Synthetic", "Learning-Curve",
          "Ablation", "Result", "Task")))
  figure_labels = _find_labels(text, "fig")
  table_labels = _find_labels(text, "tab")
  excerpts = {
      "decoder_lower_prefix_stop_gradient": _context_lines(
          text, "lower-level decoder inputs are detached", before=2, after=5),
      "predictor_prefix_stop_gradient": _context_lines(
          text, "predictor prefix", before=2, after=5),
      "dynamics_target_stop_gradient": _context_lines(
          text, "target SG", before=2, after=5),
      "training_regimes": _context_lines(
          text, "We evaluate three regimes", before=1, after=4),
      "paper_default_training_regime": _context_lines(
          text, "two-phase regime is a conservative default", before=2, after=2),
  }
  code_contract = {
      "decoder_prefix_stop_gradient": {
          "paper_status": "explicit_in_manuscript",
          "code_value": True,
          "code_location": "dreamerv3/hts.py::_nested_recon",
      },
      "predictor_prefix_stop_gradient": {
          "paper_status": "not_explicit_in_manuscript",
          "code_value": True,
          "code_location": "dreamerv3/hts.py::_sparse_dynamics",
          "required_action": "ablation or manuscript update before final claims",
      },
      "dynamics_target_stop_gradient": {
          "paper_status": "not_explicit_in_manuscript",
          "code_value": True,
          "code_location": "dreamerv3/hts.py::_sparse_dynamics",
          "required_action": "ablation or manuscript update before final claims",
      },
      "paper_default_training_regime": "two-phase tuning",
      "current_official_port_default_regime": "joint unless config overrides training_regime",
  }
  lines = [
      "# Paper Contract Source V4",
      "",
      f"- canonical source path: `{PAPER}`",
      f"- SHA256: `{sha}`",
      f"- review timestamp: `{datetime.now(timezone.utc).isoformat()}`",
      f"- hash matches expected current paper: `{sha == PAPER_HASH}`",
      "",
      "## Method Headings",
      *[f"- line {item['line']}: {item['heading']}" for item in method_headings],
      "",
      "## Experiment Headings",
      *[f"- line {item['line']}: {item['heading']}" for item in experiment_headings],
      "",
      "## Required Figure Labels",
      *[f"- line {item['line']}: `{item['label']}`" for item in figure_labels],
      "",
      "## Required Table Labels",
      *[f"- line {item['line']}: `{item['label']}`" for item in table_labels],
      "",
      "## Exact Code-Manuscript Contract",
      "```json",
      json.dumps(code_contract, indent=2, sort_keys=True),
      "```",
      "",
      "## Exact Manuscript Lines For Ambiguous Contracts",
  ]
  for key, matches in excerpts.items():
    lines += ["", f"### {key}"]
    if not matches:
      lines.append("not_explicit_in_manuscript")
    for match in matches:
      lines += ["```text", match["excerpt"], "```"]
  _write(OUT / "paper_contract_source_v4.md", "\n".join(lines) + "\n")
  return {"sha256": sha, "figure_labels": figure_labels, "table_labels": table_labels}


class Ratio:

  def __init__(self, ratio):
    self.ratio = ratio
    self.prev = None

  def __call__(self, step):
    step = int(step)
    if self.ratio == 0:
      return 0
    if self.ratio < 0:
      return 1
    if self.prev is None:
      self.prev = step
      return 1
    repeats = int((step - self.prev) * self.ratio)
    self.prev += repeats / self.ratio
    return repeats


def replay_reports():
  root = OUT / "replay_consistency_v4"
  train_ratio = 256
  batch_size = 16
  batch_length = 64
  minibatch_steps = batch_size * batch_length
  expected = train_ratio / minibatch_steps
  scheduler = Ratio(expected)
  actions = 100000
  updates = 0
  event_count = 0
  windows = {}
  cumulative = []
  for step in range(1, actions + 1):
    requested = scheduler(step)
    if requested:
      event_count += 1
    updates += requested
    cumulative.append(updates)
  for size in [100, 500, 1000, 5000]:
    got = cumulative[size - 1] / size
    windows[str(size)] = {
        "updates_per_agent_action": got,
        "absolute_error": abs(got - expected),
        "relative_error": abs(got - expected) / expected,
        "strict_status": (
            "pass" if abs(got - expected) <= 0.01 or
            abs(got - expected) / expected <= 0.05 else "fail_short_window"),
    }
  overall = updates / actions
  pure = {
      "train_ratio_replayed_steps_per_agent_action": train_ratio,
      "batch_size": batch_size,
      "batch_length": batch_length,
      "minibatch_steps": minibatch_steps,
      "expected_updates_per_agent_action": expected,
      "num_agent_actions": actions,
      "requested_update_events": event_count,
      "executed_optimizer_updates": updates,
      "updates_per_agent_action": overall,
      "absolute_error": abs(overall - expected),
      "relative_error": abs(overall - expected) / expected,
      "initial_accumulator_state": "prev=None; first call returns one update",
      "final_accumulator_state": {"prev": scheduler.prev},
      "window_diagnostics": windows,
      "strict_status": "pass" if abs(overall - expected) <= 0.01 else "fail",
  }
  _dump(root / "pure_ratio_scheduler_simulation.json", pure)
  canonical = {
      **pure,
      "real_training_trace_status": "instrumented_but_update_trace_smoke_pending",
      "canonical_expected_value_preserved": True,
      "acceptance": "blocked_until_real_update_event_trace_passes",
      "common_failure_source_audit": {
          "prefill_exclusion": "writer has is_prefill flag; real trace pending",
          "compile_only_steps": "writer has is_compile_only flag; real trace pending",
          "initial_ratio_accumulator_carry": "pure scheduler shows first-call burst",
          "resume_state": "not tested",
          "multiple_optimizer_calls_per_scheduled_update": "real trace pending",
          "counting_before_optimizer_execution": "writer logs executed update count after loop",
          "environment_repeat_semantics": "action_repeat from env config, Atari100K default repeat=4",
          "writer_numerator_denominator": "agent actions denominator; optimizer updates numerator",
      },
  }
  _dump(root / "replay_ratio_consistency_canonical_v4.json", canonical)
  md = [
      "# Replay Ratio Diagnosis V4",
      "",
      "Canonical value is unchanged: `256 / (16 * 64) = 0.25` optimizer updates per agent action.",
      "",
      "The pure `elements.when.Ratio` simulation passes asymptotically. It also exposes an initial first-call burst: when `prev=None`, the scheduler returns one update independent of ratio. Short windows can therefore deviate if accounting starts too close to the first scheduled call or fails to exclude prefill/compile-only events.",
      "",
      "Real training loop event accounting has been instrumented in `embodied/run/train.py` and writes `paper_artifacts/replay_consistency_v4/update_event_trace.jsonl`. The real trace remains pending because no accepted update-producing smoke with convergence windows has completed under V4.",
      "",
      "## Window Results",
  ]
  for key, value in windows.items():
    md.append(f"- {key} actions: {value['updates_per_agent_action']:.6f}, abs_err={value['absolute_error']:.6f}, rel_err={value['relative_error']:.6f}, status={value['strict_status']}")
  _write(root / "replay_ratio_diagnosis_v4.md", "\n".join(md) + "\n")
  return pure


def larger_flat_reports():
  root = OUT / "larger_flat_v4"
  root.mkdir(parents=True, exist_ok=True)
  target = component_matrix.ADDON_HTS_EST
  candidates = []
  for width in range(2400, 2901, 8):
    params = component_matrix.flat_mh_params(code_width=width)
    gap = abs(params - target) / target
    candidates.append({
        "candidate_width": width,
        "analytical_addon_params": params,
        "actual_addon_params": "N/A",
        "target_actual_hts_addon_params": "N/A",
        "relative_gap": gap,
        "selected": False,
        "size12m_init_verified": False,
        "forward_verified": False,
        "backward_verified": False,
        "optimizer_step_verified": False,
        "checkpoint_reload_verified": False,
    })
  selected = min(candidates, key=lambda item: item["relative_gap"])
  selected["selected"] = True
  with (root / "larger_flat_param_search_candidates_v4.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(candidates[0].keys()))
    writer.writeheader()
    writer.writerows(candidates)
  md = [
      "# Larger Flat Param Search V4",
      "",
      "`larger_flat_param` is defined as widened `flat_mh`: one dense flat code, six horizons `[1,2,4,8,16,32]`, six predictor heads, same action-window encoder contract, sparse-dynamics loss enabled, no hierarchy, no TopK, no InfoNCE, no VC.",
      "",
      f"- analytical HTS add-on target: `{target}`",
      f"- analytical selected width: `{selected['candidate_width']}`",
      f"- analytical relative gap: `{selected['relative_gap']:.6f}`",
      "- actual initialized size12m count: `pending`",
      "- checkpoint reload smoke: `pending`",
      "",
      "Paper fairness tables may not use this row until `actual_addon_params` and checkpoint reload smoke are populated.",
  ]
  _write(root / "larger_flat_param_search_report_v4.md", "\n".join(md) + "\n")
  count_md = [
      "# Actual Initialized Parameter Count Report V4",
      "",
      "- Dreamer anchor size12m Alien initialized total params: `10498772`",
      "- HTS full actual add-on params: `pending_size12m_initialization`",
      "- larger_flat_param actual add-on params: `pending_size12m_initialization`",
      "- larger_flat_param actual relative gap: `pending`",
      "",
      "Only initialized counts may be used in paper fairness tables. Analytical estimates are development diagnostics.",
  ]
  _write(root / "actual_initialized_parameter_count_report_v4.md", "\n".join(count_md) + "\n")
  _write(OUT / "actual_initialized_parameter_count_report_v4.md", "\n".join(count_md) + "\n")
  return selected


def gradient_balance_reports():
  root = OUT / "gradient_balance_v4"
  losses = {
      "L_hier": {
          "reduction": "mean over feature, batch, time, then weighted mean over levels",
          "effective_denominator": "B*T*feat_dim per decoder plus level averaging",
      },
      "L_sdyn": {
          "reduction": "mean over feature, masked mean over valid batch-time, weighted mean over levels",
          "effective_denominator": "valid_windows*feat_dim per level plus level weights",
      },
      "L_temp": {
          "reduction": "masked InfoNCE mean over valid positive pairs",
          "effective_denominator": "valid_positive_pairs",
      },
      "L_vc": {
          "reduction": "projection flattened over batch*time; mean variance and mean off-diagonal covariance",
          "effective_denominator": "projection_dim and projection_dim*(projection_dim-1)",
      },
      "L_sparse": {
          "reduction": "mean abs over batch, time, features, then mean over levels",
          "effective_denominator": "B*T*head_dim*levels",
      },
  }
  groups = [
      "dreamer_backbone", "hts_trunk",
      *[f"head_{idx}" for idx in range(1, 7)],
      *[f"decoder_{idx}" for idx in range(1, 7)],
      *[f"predictor_{idx}" for idx in range(1, 7)],
      "projector",
  ]
  diagnostics = {
      "status": "structural_audit_pending_real_autodiff",
      "source": "code reduction audit; per-loss same-batch autodiff not yet executed",
      "losses": {name: {"groups": {group: "pending" for group in groups}, **meta}
                 for name, meta in losses.items()},
  }
  _dump(root / "per_loss_gradient_norms_v4.json", diagnostics)
  md = [
      "# Loss Scale Audit V4",
      "",
      "Current status: structural reduction audit complete; real per-loss same-batch gradient norms remain pending.",
      "",
      "Prior debug trace showed temporal and VC terms larger than hierarchy/sdyn terms. No paper-default weights were changed in this package.",
      "",
      "| Loss | Reduction | Effective Denominator |",
      "| --- | --- | --- |",
  ]
  for name, meta in losses.items():
    md.append(f"| `{name}` | {meta['reduction']} | {meta['effective_denominator']} |")
  _write(root / "loss_scale_audit_v4.md", "\n".join(md) + "\n")


def smoke_and_synthetic_reports():
  smoke = {
      "status": "partial",
      "rows": [
          {"config_name": name, "forward_verified": name != "larger_flat_flops",
           "backward_verified": False, "optimizer_step_verified": False,
           "checkpoint_save_verified": name != "larger_flat_flops",
           "checkpoint_reload_verified": False}
          for name in CONFIG_NAMES
      ],
      "blocker": "full P0 optimizer/checkpoint reload smoke not yet automated",
  }
  root = OUT / "baseline_smoke_v4"
  _dump(root / "baseline_forward_backward_checkpoint_smoke_report_v4.json", smoke)
  md = ["# Baseline Forward/Backward/Checkpoint Smoke V4", "", f"Status: `{smoke['status']}`", ""]
  md += ["| config | forward | backward | optimizer | ckpt_save | ckpt_reload |",
         "| --- | --- | --- | --- | --- | --- |"]
  for row in smoke["rows"]:
    md.append(
        f"| {row['config_name']} | {row['forward_verified']} | {row['backward_verified']} | "
        f"{row['optimizer_step_verified']} | {row['checkpoint_save_verified']} | "
        f"{row['checkpoint_reload_verified']} |")
  _write(root / "baseline_forward_backward_checkpoint_smoke_report_v4.md", "\n".join(md) + "\n")

  syn = OUT / "synthetic_v4"
  sample_manifest = REPO_ROOT / "artifacts/paper_development/synthetic_multiscale_sample/manifest.json"
  if sample_manifest.exists():
    data = json.loads(sample_manifest.read_text())
    data["v4_label"] = "synthetic_multiscale_smoke"
  else:
    data = {"status": "missing_local_sample_manifest", "v4_label": "synthetic_multiscale_smoke"}
  _dump(syn / "synthetic_dataset_manifest_smoke_v4.json", data)
  _dump(syn / "synthetic_dataset_manifest_full_v4.json", {
      "status": "not_generated",
      "planned_split_trajectories": {"train": 10000, "validation": 2000, "test": 2000},
      "blocker": "full synthetic dataset generation/trainer not wired in official port",
  })
  _dump(syn / "synthetic_checkpoint_evaluator_sample_v4.json", {
      "status": "structural_placeholder",
      "model_derived_metrics": False,
      "usable_for_paper": False,
      "blocker": "connect evaluator to real HTS checkpoints",
  })

  regime = OUT / "training_regimes_v4"
  _dump(regime / "training_regime_parameter_delta_report_v4.json", {
      "status": "xfail",
      "implemented_config_names": ["joint", "detach_hts_anchor", "posthoc_frozen_backbone", "two_phase"],
      "tested_parameter_delta": False,
      "blocker": "one-step optimizer delta by module/regime not automated",
  })

  periodic = OUT / "periodic_final_eval_smoke_artifacts_v4"
  periodic.mkdir(parents=True, exist_ok=True)
  _dump(periodic / "README.json", {
      "status": "placeholder",
      "blocker": "periodic/final eval smoke needs rerun after replay consistency gate",
  })


def remaining_xfail():
  rows = [
      ("Gate A1", "blocked", "real replay update_event_trace convergence and UT-12/13B not complete"),
      ("Gate A2", "blocked", "actual size12m HTS/larger-flat counts and full P0 optimizer/checkpoint reload smoke pending"),
      ("Gate B", "blocked", "real synthetic trainer/evaluator and checkpoint-derived metrics pending"),
      ("larger_flat_flops", "P1", "FLOPs estimator/search not implemented"),
      ("DMC/DMC-GB2", "P1", "not wired in official port"),
      ("MiniHack/KeyCorridor", "P1", "THICK-compatible wrapper not implemented"),
  ]
  lines = ["# Remaining XFAIL V4", "", "| Item | Status | Reason |", "| --- | --- | --- |"]
  lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
  _write(OUT / "remaining_xfail_v4.md", "\n".join(lines) + "\n")


def main():
  if _paper_hash() != PAPER_HASH:
    raise SystemExit("paper.txt hash changed; regenerate contract expectations first")
  contract = contract_note()
  component_matrix.write(OUT)
  pure = replay_reports()
  selected = larger_flat_reports()
  gradient_balance_reports()
  smoke_and_synthetic_reports()
  remaining_xfail()
  summary = {
      "paper_sha256": contract["sha256"],
      "gate_a1": "blocked",
      "gate_a2": "blocked",
      "gate_b": "blocked",
      "canonical_expected_updates_per_agent_action": pure["expected_updates_per_agent_action"],
      "pure_scheduler_status": pure["strict_status"],
      "component_matrix_row_count": len(component_matrix.rows()),
      "component_matrix_config_names": [row["config_name"] for row in component_matrix.rows()],
      "larger_flat_selected_width_analytical": selected["candidate_width"],
      "larger_flat_actual_gap": "pending",
  }
  _dump(OUT / "v4_package_summary.json", summary)
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
