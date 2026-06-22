# Split-Branch Revision Proposal V14

No gradient-isolated reconstruction-routing candidate passed Gate D1. Do not implement this architecture before review.

## Evidence Motivating Split
V13 showed boundary degradation already at z1 under hierarchy pressure. V14 tests whether blocking hierarchy gradients into the shared trunk is sufficient. If no candidate passes, the remaining evidence points to a need for separated coarse event and fine reconstruction pathways.

## Minimal Module Graph
shared anchor h -> coarse temporal/event trunk -> z1, temporal/sdyn/event-sensitive objectives
shared anchor h -> fine reconstruction trunk -> z2..z6, nested/fine reconstruction objectives

## Parameter-Count Impact
Naively splitting the trunk increases parameters. A paper-ready version needs an equal-parameter control by reducing branch widths or matching total dense-equivalent parameter count.

## Equal-Parameter Control
Compare split-branch HTS against a single-trunk HTS with the same total parameter count and identical optimizer/update budget.

## Required Ablations
coarse branch only, fine branch only, no reconstruction gradient into coarse branch, no temporal branch, equal-param flat control.

## New Reviewer Concerns
Reviewers may attribute gains to capacity or explicit pathway engineering rather than hierarchy. The equal-param and route-disabled controls are mandatory.

## Expected Synthetic Tests
Repeat prefix gain, boundary AUPRC by prefix, factor probes, level-horizon specialization, collapse, and gradient path audits.

## Expected Atari Development Gate
Only after Synthetic Gate D1 passes should the six-game Atari Gate D2 manifest be prepared.
