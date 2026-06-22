# Telemetry Semantics Audit V10

Status: `pass`

V9 process GPU memory is renamed semantically to `gpu_process_memory_peak_mb`; PyTorch allocated/reserved fields are not populated because the runs are JAX, not PyTorch.

Startup/checkpoint timing fields are marked not_measured and must not be used for paper-final efficiency.
