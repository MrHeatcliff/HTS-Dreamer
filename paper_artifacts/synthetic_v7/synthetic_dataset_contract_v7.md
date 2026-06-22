# Synthetic Multi-Timescale Dataset Contract V7

Observation dimension: 44 = one_hot(f_fast,8) + one_hot(f_mid,8) + one_hot(f_slow,8) + one_hot(f_context,4) + one_hot(f_nuisance,16), plus Gaussian noise sigma=0.01.

Actions are signed values in {-2,-1,0,+1,+2}.

Transition rules:
- f_fast in Z_8 updates every step by signed action.
- f_mid in Z_8 accumulates signed actions over each 4-step block and updates at block boundaries.
- f_slow in Z_8 accumulates signed actions over each 16-step block and updates at block boundaries.
- f_context in Z_4 increments autonomously every 64 steps.
- f_nuisance in Z_16 increments or decrements autonomously every 1 or 2 steps, independent of actions.

Evaluation-only labels are stored in NPZ files but are excluded from trainer inputs, HTS inputs, actor/critic inputs, temporal positive sampler inputs, far-negative sampler decisions, and loss assembly.
