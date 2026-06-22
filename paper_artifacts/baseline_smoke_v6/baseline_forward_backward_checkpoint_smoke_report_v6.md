# Baseline Forward/Backward/Checkpoint Smoke V6

Status: `pass`

| config | forward | active losses | backward | opt step | reload |
| --- | --- | --- | --- | --- | --- |
| dreamer_anchor | True |  | True | True | True |
| hts_full | True | hts_hier,hts_sdyn,hts_sparse,hts_temp,hts_vc | True | True | True |
| flat_sae | True | hts_hier,hts_sparse | True | True | True |
| flat_mh | True | hts_sdyn | True | True | True |
| flat_partition_dim_matched | True | hts_hier | True | True | True |
| sgf_style_flat_same_code | True | hts_sdyn,hts_vc | True | True | True |
| recon_only_hierarchy | True | hts_hier | True | True | True |
| matryoshka_only | True | hts_hier,hts_sparse | True | True | True |
| dense_multistride_no_sparse | True | hts_hier,hts_sdyn,hts_temp,hts_vc | True | True | True |
| larger_flat_param | True | hts_sdyn | True | True | True |
| hts_no_temp | True | hts_hier,hts_sdyn,hts_sparse,hts_vc | True | True | True |
| hts_no_vc | True | hts_hier,hts_sdyn,hts_sparse,hts_temp | True | True | True |
| hts_no_hier | True | hts_sdyn,hts_sparse,hts_temp,hts_vc | True | True | True |
| hts_no_sdyn | True | hts_hier,hts_sparse,hts_temp,hts_vc | True | True | True |
