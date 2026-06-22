# Training Regime Parameter Delta Report V6

Status: `pass`

| regime | dreamer delta | hts trunk delta | actor delta | critic delta | assertions |
| --- | ---: | ---: | ---: | ---: | --- |
| joint | 0.1 | 0.2 | 0.05 | 0.05 | True |
| detach_hts_anchor | 0.0 | 0.2 | 0.05 | 0.05 | True |
| posthoc_frozen_backbone | 0.0 | 0.2 | 0.0 | 0.0 | True |
| two_phase_phase1 | 0.0 | 0.2 | 0.05 | 0.05 | True |
| two_phase_phase2 | 0.1 | 0.2 | 0.05 | 0.05 | True |
