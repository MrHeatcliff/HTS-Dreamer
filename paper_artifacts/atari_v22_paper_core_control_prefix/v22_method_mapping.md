# V22 Method Mapping

Method name: `hts_paper_core_control_prefix`

Short alias/config: `hts_paper_core_zfull`

Implementation files:

| Paper component | Implementation |
| --- | --- |
| Sparse prefix decomposition | `dreamerv3/hts.py::HTSAux._encode`, level-wise TopK |
| Nested prefix reconstruction | `dreamerv3/hts.py::HTSAux._nested_recon(..., beta_schedule="front")` |
| Multi-stride prefix dynamics | `dreamerv3/hts.py::HTSAux._prefix_dynamics` |
| Control-aware prefix objective | `dreamerv3/hts.py::HTSAux._control_prefix` |
| Actor/critic on `z_full` | `dreamerv3/hts_agent.py::_ac_input`, `policy`, imagination and replay value paths |

Default objective:

```text
Lmodel = LWM + lambda_hier Lhier + lambda_sdyn Lsdyn + lambda_ctrl Lctrl
```

Default disabled stabilizers:

```text
Ltemp = 0
Lvc = 0
Lsparse = 0
Lred = 0
```

User-selected V22 sparse contract after ablation:

```text
levels = 4
head_dim = 32
topk_per_level = [8, 8, 8, 8]
strides = [32, 8, 2, 1]
vicreg_cov_scale = 0.001  # logged/configured; VC loss disabled by default
actor_critic_input = z_full
```
