# Gate-D Proposed Plan V8

Do not run in V8.

## Synthetic full fixed-buffer
Methods: hts_full, flat_sae, flat_mh, flat_partition_dim_matched, matryoshka_only, dense_multistride_no_sparse, hts_no_temp, hts_no_vc, hts_no_hier, hts_no_sdyn.
Seeds: [0,1,2,3,4].

## Atari six-game development subset
Tasks: Alien, Asterix, Breakout, Hero, MsPacman, Seaquest.
Methods: dreamer_anchor, hts_full, flat_mh, larger_flat_param, matryoshka_only, dense_multistride_no_sparse, hts_no_temp, hts_no_sdyn.
Seeds: [0,1,2].

## KeyCorridor wiring checklist
N = [4,8,11], seeds=[0,1,2]. Install/wire MiniHack only after review.
