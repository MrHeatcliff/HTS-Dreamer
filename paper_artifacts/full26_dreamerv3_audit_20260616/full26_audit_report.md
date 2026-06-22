# DreamerV3 Official Full26 Audit

Generated: 2026-06-16 Asia/Ho_Chi_Minh

Target: `440000` raw frames; completed if progress >= `435600` frames.

Completion is based on max progress from update trace, train metrics, metrics, and score files.

## Summary

- Completed seeds: `118/130`
- Problem seeds: `12`
- Games present on disk: `25/26`
- Missing game dirs: `up_n_down`

## Problem Seeds

| game | seed | status | progress frames | latest score | progress source | error hint |
| --- | ---: | --- | ---: | ---: | --- | --- |
| breakout | 3 | partial | 218760 | 0.0 | update_trace_v6 | interrupted/launcher stopped before target; no stdout traceback found |
| road_runner | 4 | partial | 6252 | 0.0 | update_trace_v6 | interrupted/launcher stopped before target; no stdout traceback found |
| seaquest | 0 | partial | 42480 | 340.0 | update_trace_v6 | interrupted/launcher stopped before target; no stdout traceback found |
| seaquest | 1 | partial | 121280 | 660.0 | update_trace_v6 | interrupted/launcher stopped before target; no stdout traceback found |
| seaquest | 2 | missing |  |  |  | not launched |
| seaquest | 3 | missing |  |  |  | not launched |
| seaquest | 4 | missing |  |  |  | not launched |
| up_n_down | 0 | missing |  |  |  | not launched |
| up_n_down | 1 | missing |  |  |  | not launched |
| up_n_down | 2 | missing |  |  |  | not launched |
| up_n_down | 3 | missing |  |  |  | not launched |
| up_n_down | 4 | missing |  |  |  | not launched |

## Per-Game Summary

| game | completed | partial | started_no_progress | missing | problem seeds |
| --- | ---: | ---: | ---: | ---: | --- |
| alien | 5 | 0 | 0 | 0 |  |
| amidar | 5 | 0 | 0 | 0 |  |
| assault | 5 | 0 | 0 | 0 |  |
| asterix | 5 | 0 | 0 | 0 |  |
| bank_heist | 5 | 0 | 0 | 0 |  |
| battle_zone | 5 | 0 | 0 | 0 |  |
| boxing | 5 | 0 | 0 | 0 |  |
| breakout | 4 | 1 | 0 | 0 | 3 |
| chopper_command | 5 | 0 | 0 | 0 |  |
| crazy_climber | 5 | 0 | 0 | 0 |  |
| demon_attack | 5 | 0 | 0 | 0 |  |
| freeway | 5 | 0 | 0 | 0 |  |
| frostbite | 5 | 0 | 0 | 0 |  |
| gopher | 5 | 0 | 0 | 0 |  |
| hero | 5 | 0 | 0 | 0 |  |
| james_bond | 5 | 0 | 0 | 0 |  |
| kangaroo | 5 | 0 | 0 | 0 |  |
| krull | 5 | 0 | 0 | 0 |  |
| kung_fu_master | 5 | 0 | 0 | 0 |  |
| ms_pacman | 5 | 0 | 0 | 0 |  |
| pong | 5 | 0 | 0 | 0 |  |
| private_eye | 5 | 0 | 0 | 0 |  |
| qbert | 5 | 0 | 0 | 0 |  |
| road_runner | 4 | 1 | 0 | 0 | 4 |
| seaquest | 0 | 2 | 0 | 3 | 0,1,2,3,4 |
| up_n_down | 0 | 0 | 0 | 5 | 0,1,2,3,4 |
