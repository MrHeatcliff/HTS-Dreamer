# Replay Ratio Diagnosis V6

Status: `pass`

- Total events: `6100`
- Post-prefill events: `5013`
- Action repeat: `4`
- Batch shape: `16 x 64`
- Expected updates/action: `0.25`
- Command meta: `{'command': 'v6 replay smoke size1m atari100k_pong steps6100', 'exit_status': 0, 'wall_clock_seconds': 223}`

| Window | Executed Updates | Realized Updates/Action | Abs Error | Rel Error | Pass |
| --- | ---: | ---: | ---: | ---: | --- |
| 100 | 25 | 0.250000 | 0.000000 | 0.000000 | True |
| 500 | 125 | 0.250000 | 0.000000 | 0.000000 | True |
| 1000 | 250 | 0.250000 | 0.000000 | 0.000000 | True |
| 5000 | 1250 | 0.250000 | 0.000000 | 0.000000 | True |
