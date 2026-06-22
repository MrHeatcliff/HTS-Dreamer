# Drift Toggle Results Seed0 V17

Observation: V15 exact shared-trunk training is low while historical direct-head lineage is high.

Hypothesis: the decisive drift is protocol/parameterization lineage, centered on direct-head historical recipe versus V15 shared trunk.

Minimal diagnostic: replay historical seed0 and re-evaluate historical checkpoint under current evaluator.

Expected evidence: high AUPRC for both historical replay and historical checkpoint, low AUPRC for V15 exact.

Decision: candidate drift factor identified; replay seed0 overall AUPRC `0.733841`.
