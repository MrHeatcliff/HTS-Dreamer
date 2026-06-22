# Paired Initialization and Sampler Audit V13

Status: `pass`

For each seed, synthetic runs use `init_params(seed)` and `np.random.default_rng(seed)` for sampler order. First 100 sampled windows match across compared configs.
