# Paper Contract Source

canonical manuscript path: `/mnt/disk1/backup_user/dat.tt2/xuance/paper.txt`

SHA256 hash: `9c511de120655c2c6038ce46c4c613dab9e9c3ae83db01a4b86126745b0276a6`

date reviewed: 2026-06-10 Asia/Ho_Chi_Minh

## Method Section Headings

- `\section{Method}`
- `\subsection{Hierarchical Sparse Decomposition}`
- `\subsection{Multi-Stride Sparse Dynamics}`
- `\subsection{Temporal Consistency and Anti-Collapse}`
- `\subsection{Sparsity}`
- `\subsection{Full Objective and Training Regimes}`

## Experiment Section Headings

- `\section{Experimental Evaluation}`
- `\subsection{Research Questions}`
- `\subsection{Experimental Setup}`
- `\subsection{Benchmark Suites}`
- `\subsection{Evaluation Metrics}`
- `\subsection{Statistical Reporting and Fairness Rules}`
- `\subsection{Baselines and Fairness}`
- `\subsection{Main Control Utility}`
- `\subsection{Prefix Refinement and Temporal Specialization}`
- `\subsection{Boundary Responsiveness, Revisitation, and Robustness}`
- `\subsection{Ablations}`
- `\subsection{Collapse and Feature Usage}`
- `\subsection{Compute and Fairness}`
- `\subsection{Detailed Learning Curves and Reliable Aggregation}`
- `\subsection{Benchmark-Level Result Tables}`

## Required Figure Labels

- `fig:keycorridor-learning`: present
- `fig:keycorridor-milestones`: present
- `fig:synthetic-training`: present
- `fig:hts-ablation-learning`: present
- `fig:atari100k-curves`: present
- `fig:dmc-visual-curves`: present
- `fig:dmcgb2-learning`: present
- `fig:level-horizon`: present

## Required Table Labels Mentioned By Current Contract

- `tab:prefix`: present
- `tab:level-horizon`: present
- `tab:collapse`: present
- `tab:temporal-robustness`: present
- `tab:main-results`: present
- `tab:ablation-plan`: present
- `tab:compute`: present
- `tab:matched-controls`: present
- `tab:atari-task-results`: present
- `tab:dmc-task-results`: present

## Review Note

`paper.txt` is the canonical current manuscript contract for this development cycle. The `.tex` source may be added later for compilation, but it is not required for Gate A2/Gate B implementation work.

## Code-Manuscript Contract Check

- latent anchor: `rssm_repfeat`
- latent anchor source: `dreamerv3.rssm.RSSM.loss`
- number of levels: `6`
- head dimension: `32`
- total dictionary width: `192`
- TopK budgets: `[8, 8, 8, 8, 8, 8]`, total sparse active budget `48`
- coarse-to-fine strides: `[32, 16, 8, 4, 2, 1]`
- prefix stop-gradient: lower prefixes are stopped for decoder/predictor level `ell`
- temporal objective: masked InfoNCE over projected coarse code `z^(1)`
- positive-window K: `4`
- temperature tau: `0.1`
- far-negative default: `none`; supported modes are `none`, `hard`, `soft`
- training regime default: `joint`
- loss weights in current config: `l_hier=0.1`, `l_sdyn=0.1`, `l_temp=0.01`, `l_vc=0.01`, `l_sparse=1e-5`
- per-level reconstruction weights: uniform `1/6`
- per-level sparse-dynamics weights: uniform `1/6`

Status: aligned with the current official-code HTS config, but this note must be regenerated after the canonical `.tex` source is copied into the repo.
