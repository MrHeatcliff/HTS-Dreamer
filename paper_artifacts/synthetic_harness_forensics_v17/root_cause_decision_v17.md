# Root Cause Decision V17

Decision: `HARNESS_DRIFT_FIXED_PROTOCOL_LOCKED`

- Observation: historical checkpoints and replay remain high under current evaluator.
- Hypothesis: V15 low baseline is training harness/protocol drift, centered on direct-head historical recipe versus V15 shared-trunk exact harness.
- Minimal diagnostic: checkpoint re-eval plus V9/V12 replay.
- Expected evidence: historical high, V15 exact low, replay high.
- Decision: lock historical synthetic protocol and rerun Gate D1 later under that protocol.
