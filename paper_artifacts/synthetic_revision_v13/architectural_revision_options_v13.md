# Architectural Revision Options V13

No coarse-protected per-level weighting candidate passed the mechanism-preservation gate. Do not implement these options without review.

## Option A: Separate Coarse Temporal/Event Path From Reconstruction Path
Claim addressed: preserve slow/event features while allowing reconstruction refinement.
Code changes required: split z1 or add a parallel event-preserving branch before nested reconstruction.
New confounds introduced: extra capacity and path-specific losses.
Required ablations: equal-parameter split, no-event branch, reconstruction-only branch.
Expected reviewer concern: improvement may come from added capacity rather than hierarchy.

## Option B: Stop Hierarchy Reconstruction Gradient Into z1
Claim addressed: protect coarse code from reconstruction pressure.
Code changes required: detach z1 for hierarchy decoder gradients while preserving temporal/sdyn gradients.
New confounds introduced: z1 may become underconstrained for reconstruction.
Required ablations: detach z1 only, detach z1:z2, detach decoder input only.
Expected reviewer concern: hierarchy reconstruction claim weakens for coarse code.

## Option C: Apply Hierarchy Reconstruction Only To Residual Fine Heads z2..z6
Claim addressed: let fine heads absorb reconstruction without corrupting coarse event features.
Code changes required: remove or downweight D1 and reconstruct residual targets for later levels.
New confounds introduced: changes objective semantics from nested reconstruction to residual reconstruction.
Required ablations: residual-only, nested-only, hybrid.
Expected reviewer concern: less direct comparison to Matryoshka-style hierarchy.

## Option D: Add Explicit Event/Boundary Auxiliary Objective
Claim addressed: maintain event-sensitive boundaries while strengthening reconstruction.
Code changes required: boundary/event prediction head and loss.
New confounds introduced: uses additional supervision or pseudo-label assumptions.
Required ablations: event loss only, event + hierarchy, no temporal contrastive.
Expected reviewer concern: paper claim may rely on event labels rather than unsupervised structure.
