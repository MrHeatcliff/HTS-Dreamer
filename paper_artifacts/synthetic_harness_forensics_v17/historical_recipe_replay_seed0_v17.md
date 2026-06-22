# Historical Recipe Replay Seed0 V17

Observation: V9/V12 historical recipe was replayed from the V9 250-update checkpoint.

Hypothesis: if protocol drift is in V15 harness, the historical recipe should remain high under the current evaluator.

Minimal diagnostic: replay seed 0 to update 1000 and evaluate with V15 evaluator.

Expected evidence: AUPRC within 0.05 of recorded historical seed0 and still high.

Decision: reproduces_high_boundary=`true`; overall AUPRC=`0.733841`, macro=`0.646358`, historical=`0.733841`.
