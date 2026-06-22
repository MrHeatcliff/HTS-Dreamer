# Prefix Reconstruction Criterion V10

Target tensor: synthetic observation vector `obs` from the test split of `synthetic_multiscale_full_v7`.

Normalization: `NRMSE = RMSE(prediction, obs) / sqrt(mean(obs^2) + 1e-8)` for the current evaluation batch.

Prefix levels: levels 1..6. Decoder `D_l` receives concatenated codes `z^(1)..z^(l)` and reconstructs the original observation tensor.

Checkpoint used for acceptance: final checkpoint unless explicitly marked as checkpoint-trajectory diagnostic.

Lower is better. Tolerance epsilon: `1e-06`.

Hard diagnostic criterion: strict monotonicity for every seed, all marginal gains > epsilon.

Paper-claim candidate criterion: aggregate mean improves with prefix depth and end-to-end gain is positive with uncertainty reported. This is reported separately and does not convert Gate D1 to pass in V10.
