# Replay Ratio Diagnosis V4

Canonical value is unchanged: `256 / (16 * 64) = 0.25` optimizer updates per agent action.

The pure `elements.when.Ratio` simulation passes asymptotically. It also exposes an initial first-call burst: when `prev=None`, the scheduler returns one update independent of ratio. Short windows can therefore deviate if accounting starts too close to the first scheduled call or fails to exclude prefill/compile-only events.

Real training loop event accounting has been instrumented in `embodied/run/train.py` and writes `paper_artifacts/replay_consistency_v4/update_event_trace.jsonl`. The real trace remains pending because no accepted update-producing smoke with convergence windows has completed under V4.

## Window Results
- 100 actions: 0.250000, abs_err=0.000000, rel_err=0.000000, status=pass
- 500 actions: 0.250000, abs_err=0.000000, rel_err=0.000000, status=pass
- 1000 actions: 0.250000, abs_err=0.000000, rel_err=0.000000, status=pass
- 5000 actions: 0.250000, abs_err=0.000000, rel_err=0.000000, status=pass
