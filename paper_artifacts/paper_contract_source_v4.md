# Paper Contract Source V4

- canonical source path: `/mnt/disk1/backup_user/dat.tt2/xuance/paper.txt`
- SHA256: `9c511de120655c2c6038ce46c4c613dab9e9c3ae83db01a4b86126745b0276a6`
- review timestamp: `2026-06-11T04:49:21.494417+00:00`
- hash matches expected current paper: `True`

## Method Headings
- line 278: Multi-Stride Sparse Dynamics
- line 308: Temporal Consistency and Anti-Collapse
- line 360: Sparsity
- line 381: Full Objective and Training Regimes

## Experiment Headings
- line 432: Experimental Evaluation
- line 444: Experimental Setup
- line 514: Benchmark Suites
- line 869: Hero Result Figure: Breadth, Signature Challenge, and Structure
- line 1101: Ablations
- line 1404: DreamerV3-Style Backbone Ablations
- line 1879: Benchmark-Level Result Tables

## Required Figure Labels
- line 155: `fig:overview`
- line 902: `fig:hero-results`
- line 945: `fig:horizon-sweep`
- line 960: `fig:rliable-summary`
- line 1015: `fig:level-horizon`
- line 1030: `fig:prefix-refinement`
- line 1082: `fig:spliced-trajectory`
- line 1098: `fig:nuisance-event`
- line 1166: `fig:collapse-dashboard`
- line 1209: `fig:compute-pareto`
- line 1228: `fig:loss-interactions`
- line 1245: `fig:prefix-rollouts`
- line 1261: `fig:factor-recovery`
- line 1450: `fig:backbone-audit`
- line 1662: `fig:scaling-transfer`
- line 1704: `fig:openloop-rollouts`
- line 1765: `fig:keycorridor-learning`
- line 1783: `fig:keycorridor-milestones`
- line 1801: `fig:synthetic-training`
- line 1819: `fig:hts-ablation-learning`
- line 1836: `fig:atari100k-curves`
- line 1851: `fig:dmc-visual-curves`
- line 1869: `fig:dmcgb2-learning`
- line 1965: `fig:second-backbone`

## Required Table Labels
- line 490: `tab:train-setup`
- line 594: `tab:protocol`
- line 745: `tab:metrics`
- line 791: `tab:baselines`
- line 822: `tab:baseline-execution-tiers`
- line 866: `tab:main-results`
- line 923: `tab:hero-panel-slots`
- line 980: `tab:prefix`
- line 999: `tab:level-horizon`
- line 1059: `tab:temporal-robustness`
- line 1125: `tab:ablation-plan`
- line 1147: `tab:collapse`
- line 1189: `tab:compute`
- line 1312: `tab:cross-domain-protocol`
- line 1333: `tab:backbone-reproduction`
- line 1370: `tab:claim-evidence-registry`
- line 1401: `tab:experiment-suite-matrix`
- line 1433: `tab:dreamer-backbone-audits`
- line 1485: `tab:dreamerv3-robustness-audit`
- line 1515: `tab:matched-controls`
- line 1542: `tab:scaling-grid`
- line 1587: `tab:nearest-method-matrix`
- line 1615: `tab:offline-diagnosis`
- line 1647: `tab:hyper-transfer`
- line 1689: `tab:rollout-fidelity`
- line 1746: `tab:learning-curve-artifacts`
- line 1903: `tab:dmc-task-results`
- line 1923: `tab:atari-task-results`
- line 1950: `tab:planner-audit`

## Exact Code-Manuscript Contract
```json
{
  "current_official_port_default_regime": "joint unless config overrides training_regime",
  "decoder_prefix_stop_gradient": {
    "code_location": "dreamerv3/hts.py::_nested_recon",
    "code_value": true,
    "paper_status": "explicit_in_manuscript"
  },
  "dynamics_target_stop_gradient": {
    "code_location": "dreamerv3/hts.py::_sparse_dynamics",
    "code_value": true,
    "paper_status": "not_explicit_in_manuscript",
    "required_action": "ablation or manuscript update before final claims"
  },
  "paper_default_training_regime": "two-phase tuning",
  "predictor_prefix_stop_gradient": {
    "code_location": "dreamerv3/hts.py::_sparse_dynamics",
    "code_value": true,
    "paper_status": "not_explicit_in_manuscript",
    "required_action": "ablation or manuscript update before final claims"
  }
}
```

## Exact Manuscript Lines For Ambiguous Contracts

### decoder_lower_prefix_stop_gradient
```text
261: \label{eq:lhier}
262: \end{equation}
263: By default, the lower-level decoder inputs are detached when training a finer
264: residual pathway:
265: \begin{equation}
266: D_{\ell}\!\left(
267: \operatorname{sg}(z_t^{(1)}),\ldots,
268: \operatorname{sg}(z_t^{(\ell-1)}),z_t^{(\ell)}
```

### predictor_prefix_stop_gradient
not_explicit_in_manuscript

### dynamics_target_stop_gradient
```text
299: that intervene before the target latent; it excludes $a_{t+\Delta_{\ell}}$.
300: 
301: % TODO(P0-METHOD-SDYN): Compare target SG vs. no target SG.
302: % TODO(P0-METHOD-SDYN): Add flat multi-horizon and uniform-stride controls.
303: % TODO(P0-METHOD-SDYN): Log level-by-horizon prediction error during training.
304: % TODO(P0-METHOD-ACTION): Compare state-only, current-action, raw action-subsequence,
305: % and action-summary-encoder inputs. This tests whether the hierarchy captures
306: % controllable dynamics rather than observation autocorrelation alone.
```
```text
426: % TODO(P0-METHOD-ALG): Add a second algorithm or appendix pseudocode that states
427: % all sampling details: sequence length, horizon availability mask, far-negative
428: % eligibility, target SG, and action indexing.
429: % TODO(P0-METHOD-SUPERVISION): Add a table distinguishing signals used during
430: % training, inference, and evaluation-only analysis.
431: 
432: \section{Experimental Evaluation}
433: \subsection{Research Questions}
```

### training_regimes
```text
393: \end{equation}
394: We evaluate three regimes: frozen post-hoc extraction ($\eta_{\phi}=0$),
395: two-phase tuning ($0<\eta_{\phi}\ll\eta_{\psi}$), and fully joint tuning
396: ($\eta_{\phi}=\eta_{\psi}$). The two-phase regime is a conservative default,
397: not a universal optimum.
398: 
```

### paper_default_training_regime
```text
394: We evaluate three regimes: frozen post-hoc extraction ($\eta_{\phi}=0$),
395: two-phase tuning ($0<\eta_{\phi}\ll\eta_{\psi}$), and fully joint tuning
396: ($\eta_{\phi}=\eta_{\psi}$). The two-phase regime is a conservative default,
397: not a universal optimum.
398: 
```
