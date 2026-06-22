# Stop-Gradient Contract V5

- `decoder_prefix_stop_gradient = true`: explicit in `paper.txt`; deterministic trace passes.
- `predictor_prefix_stop_gradient = false`: not explicit in `paper.txt`; V5 development default is no detach. Detached predictor-prefix is now a named code flag/ablation.
- `dynamics_target_stop_gradient = true`: code default; manuscript contains TODO for target-SG versus no-target-SG audit. Both modes are implemented by flag.

## Code Flags
- `agent.hts.decoder_prefix_stop_gradient`
- `agent.hts.predictor_prefix_stop_gradient`
- `agent.hts.dynamics_target_stop_gradient`

## Trace Summary
```json
{
  "decoder_prefix_stop_gradient": {
    "current_level_grad_norm": 2.4494895935058594,
    "default": true,
    "disabled_lower_prefix_grad_norm": 2.4494895935058594,
    "lower_prefix_grad_norm": 0.0,
    "status": "pass"
  },
  "dynamics_target_stop_gradient": {
    "default": true,
    "status": "pass",
    "target_no_sg_grad_norm": 0.125,
    "target_sg_grad_norm": 0.0883883461356163
  },
  "predictor_prefix_stop_gradient": {
    "default": false,
    "default_lower_prefix_grad_norm": 4.898979187011719,
    "detached_ablation_lower_prefix_grad_norm": 0.0,
    "paper_status": "not_explicit_in_manuscript",
    "status": "pass"
  }
}
```
