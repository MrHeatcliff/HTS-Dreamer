# Replay Ratio Diagnosis V5

Canonical ratio remains `256 / (16 * 64) = 0.25` optimizer updates per agent action.

A deterministic scheduler/accounting event trace with the V5 fields passes all required windows. This verifies numerator/denominator accounting, but a real update-producing Atari smoke is still required before Gate A1 can pass.

| Window | Realized | Abs Error | Rel Error |
| --- | --- | --- | --- |
| 100 | 0.250000 | 0.000000 | 0.000000 |
| 500 | 0.250000 | 0.000000 | 0.000000 |
| 1000 | 0.250000 | 0.000000 | 0.000000 |
| 5000 | 0.250000 | 0.000000 | 0.000000 |
