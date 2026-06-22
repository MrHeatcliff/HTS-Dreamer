# V19-AB Retry Resolution Report

Status: `resolved`

## Completed Runs

All required V19-AB runs are complete:

| game | seeds | status |
| --- | --- | --- |
| Alien | 0, 1, 2 | complete |
| Breakout | 0, 1, 2 | complete |

## Failed/Partial Runs

| run | issue | action |
| --- | --- | --- |
| `breakout/seed_0` | W&B GIF encoding crashed with `struct.error: ushort format requires 0 <= number <= 65535` near the end of training. | Re-ran as `breakout/seed_0_retry_no_video`. |
| `breakout/seed_2` | W&B video encoding hit `OSError: [Errno 28] No space left on device` before reaching the Atari100K frame target. | Re-ran as `breakout/seed_2_retry_no_video`. |

## Fix Applied

The retry launcher disables both report video generation and policy-frame episode videos:

```text
--agent.report False
--run.log_policy_video False
--run.report_every 999999
```

`run.log_policy_video` was added as a default-preserving flag. Default remains `True`; V19 retry commands set it to `False`.

## Final Latest Scores

| game | seed | latest score | logged frames | source run |
| --- | ---: | ---: | ---: | --- |
| Alien | 0 | 640 | 438668 | original |
| Alien | 1 | 1820 | 437540 | original |
| Alien | 2 | 790 | 437496 | original |
| Breakout | 0 | 11 | 438888 | retry no-video |
| Breakout | 1 | 7 | 438564 | original |
| Breakout | 2 | 9 | 439056 | retry no-video |

V19-AB decision after retry: `PASS_ATARI_DEV_SANITY_SMALL`.

Gate D2 full benchmark remains blocked unless explicitly approved later.
