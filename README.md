# cuPilot-Kernels

This repository collects the best-performing kernels discovered across four cuPilot evolution epochs.

- `kernels/` holds the top kernel artifact per problem after the fourth evolution pass. For GEMM problems the relevant PIDs span `1` through `18`.
- `latency_summary.csv` summarizes per-problem latency measurements, comparing each epoch generated kernels (Epoch0 ~ Epoch3) against the AI CUDA Engineer baseline to target-PyTorch kernels speedup trends.

