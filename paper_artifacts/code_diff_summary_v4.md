# Code Diff Summary V4

## Core HTS Port
- Added `dreamerv3/hts.py` with hierarchical sparse heads, nested reconstruction, sparse multi-stride dynamics, temporal contrastive loss, VICReg-style VC loss, sparsity metrics, and same-code comparator variants.
- Added `dreamerv3/hts_agent.py` and `dreamerv3/main_hts.py` to instantiate the HTS agent without editing the baseline `dreamerv3.main` entrypoint semantics.
- Implemented `flat_partition_dim_matched` as a trainable dense six-partition 192-dimensional bottleneck with one full-code reconstruction decoder.
- Implemented `larger_flat_param` semantics as a widened `flat_mh` comparator in the matrix; actual size12m parameter matching remains pending.

## Artifact Logging
- Added `embodied/run/paper_artifacts.py` to write paper-oriented metadata, episode rows, train summaries, HNS references, replay accounting, and run-final artifacts.
- Updated `embodied/run/train.py` to log paper artifacts and event-level replay update traces.
- Updated `embodied/run/eval_only.py` for paper evaluation artifacts.

## V4 Audit Infrastructure
- Added `dreamerv3/component_matrix.py` to generate V3 and V4 JSON/CSV/MD component matrices from one source of truth.
- Added `dreamerv3/hts_unit_tests.py` for deterministic method-contract tests and gate XFAIL reporting.
- Added `dreamerv3/hts_debug_trace.py` for one-batch static/backward traces and per-loss gradient norms.
- Added `dreamerv3/generate_v4_artifacts.py` for paper contract, replay diagnosis, matrix generation, larger-flat analytical search, gradient-balance placeholders, synthetic placeholders, and gate summaries.

## Current Guardrail
No paper-final or long experiment was launched in this V4 package. Gate A1/A2/B remain blocked until the listed XFAILs are resolved.
