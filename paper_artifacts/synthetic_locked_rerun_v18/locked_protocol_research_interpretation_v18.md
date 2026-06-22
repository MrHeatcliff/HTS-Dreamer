# Locked Protocol Research Interpretation V18

Observation: the locked protocol reproduces the historical direct-head baseline, so V18 comparisons are no longer using the drifted V15 shared-trunk harness.

Hypothesis: if hierarchy strengthening is a stable mechanism, it should improve prefix fidelity over the locked baseline while preserving boundary/event sensitivity.

Minimal test: compare `locked_hier_x3` and `locked_no_hier_loss` against `locked_baseline_direct_head`; mark V14 trunk-isolation candidate as not applicable because the locked protocol has no shared trunk.

Evidence: `locked_hier_x3` satisfies prefix and mechanism-preservation rules.

Decision: a locked synthetic development candidate exists, but Atari Gate D2 remains blocked until separately approved.

Routing note: `locked_recon_trunk_isolated_fine_only_x3` is non-equivalent to V14 because direct-head locked protocol has no trunk gradient route to isolate.
