# Historical Baseline Comparability V15

Status: `fail`

- historical baseline uses direct heads without shared trunk; V14 candidates use explicit shared trunk
- historical baseline checkpoint lineage is V9/V12 continuation; V14 candidates are fresh V14 runs
- V14 baseline_shared_trunk is a metric reference, not a V14 retraining run
- checkpoint schedules and initialization semantics differ
