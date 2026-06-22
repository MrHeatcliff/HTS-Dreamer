# V22 Decision

Observation
: Previous HTS variants were auxiliary and not sufficiently control-aware.

Hypothesis
: Routing actor/critic through sparse `z_full` plus prefix reward/continue/value losses should make the hierarchy control-aware.

Minimal experiment
: Implement `hts_paper_core_zfull`, run unit tests, synthetic control diagnostic, then Atari smoke before any Stage A run.

Evidence
: See `v22_unit_tests.*`, `v22_synthetic_control_diagnostic.*`, and `v22_atari_smoke.*` when populated.

Decision
: `V22_INCONCLUSIVE_RUN_OR_METRIC`

Next step
: Run Breakout Stage A seeds 0,1,2 from `launch_v22_breakout_stage_a.sh`.

Gate E allowed
: `False`

Expansion to more games allowed
: `False`
