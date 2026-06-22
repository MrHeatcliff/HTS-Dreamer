# Loss Reduction Audit V5

No coefficient was changed. V5 confirms that reduction semantics are explicit, but the deterministic fixture still shows temporal InfoNCE dominating projector/head-1/trunk gradients.

| Loss | Raw | Weighted | Batch/Feature/Mask Denominator | Level Denominator | Reduction |
| --- | --- | --- | --- | --- | --- |
| `hier` | `0.00010011152335209772` | `1.0011152880906593e-05` | B*T*feat_dim | levels | mean feature/time/batch then level mean |
| `sdyn` | `0.0001537101052235812` | `1.537101161375176e-05` | valid_windows*feat_dim | levels | masked mean then level mean |
| `temp` | `7.048462390899658` | `0.07048462331295013` | valid positive pairs | none | masked InfoNCE mean |
| `vc` | `0.9899988174438477` | `0.00989998783916235` | batch*time and projection_dim | none | mean variance + mean offdiag covariance |
| `sparse` | `0.0006682152743451297` | `6.682152609016612e-09` | B*T*head_dim | levels | mean abs then level mean |

## Normalization Diagnosis
- `hier` and `sdyn`: mean-reduced over feature and batch/time, then averaged by level weights; no sum bug found in code audit.
- `temp`: mean-reduced masked InfoNCE, but raw scale is naturally much larger on the deterministic fixture.
- `vc`: mean-reduced but still larger than reconstruction/dynamics on the fixture.
- `sparse`: numerically tiny under current coefficient.

Next tuning step, after real Synthetic trainer exists: small synthetic-smoke coefficient sweep.
