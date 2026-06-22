# Metric Protocol V20

X-axis: raw frames. Agent actions = raw frames / 4. Target = 440000 raw frames = 110000 agent actions.

Binning: 20 fixed bins, edges `linspace(0, 440000, 21)`, width 22000 raw frames. Rows are `episode/score` from `scores.jsonl`. Empty bins are missing; no interpolation in primary metrics.

Primary metrics: `auc_20bin_mean`, `final_20pct_mean` over frames >= 352000, and `final_bin_mean` for [418000, 440000].

Secondary only: latest episode score, best bin score, seed std/counts.
