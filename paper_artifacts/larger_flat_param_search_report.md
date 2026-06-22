# Larger Flat Parameter Search

Date: 2026-06-10 Asia/Ho_Chi_Minh

Status: analytical width selected; initialized size12m parameter delta still pending.

## Contract

- Source baseline: `flat_mh`
- Target: match HTS add-on parameter count within 2 percent.
- Matching target source: analytical estimate from current official HTS module formula.
- Do not use this as initialized parameter evidence until a size12m model is initialized and counted.

## Analytical Search Result

```text
target_addon_params = 10,158,720
selected_width = 2648
estimated_flat_mh_addon_params = 10,159,960
estimated_relative_gap = 0.0001220626
```

Candidate neighborhood:

```text
width  params    relative_gap
2616   10102584  0.0055258930
2624   10116928  0.0041139041
2632   10131272  0.0027019152
2640   10145616  0.0012899263
2648   10159960  0.0001220626
2656   10174304  0.0015340515
2664   10188648  0.0029460404
2672   10202992  0.0043580294
2680   10217336  0.0057700183
```

## Remaining Work

- Initialize `larger_flat_param` under the actual size12m Atari config.
- Export actual add-on parameter count from initialized model.
- Replace analytical `target_addon_params` with initialized HTS add-on count if the initialized count differs.
- Write machine-readable candidate CSV after actual initialization search.
