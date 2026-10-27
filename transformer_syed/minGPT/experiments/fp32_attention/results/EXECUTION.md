# Execution provenance

- A100 probe requested on partition quick, account yli15, with `--gres=gpu:A100:1`: job 2414206.
- Scheduler reported no start estimate and reason: "Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions".
- Cancellation raced allocation: accounting shows one second on gpu-4-11 before cancellation, with only environment startup in its log and no experiment output. Used the user-authorized A30 fallback.
- A30 correctness probe: job 2414214, account isp / partition academic; completed successfully.
- Main A30 experiment: job 2414225, account isp / partition academic, gpu-4-23.
- All main variants share one GPU allocation. Tests set PyTorch CPU intra-op threads to 1; this remains the timing setting, within an allocation of 4 CPUs.
- The custom kernel is inference-only. Its compiled dot products are checked for FP32 FMA and absence of Tensor Core MMA/TF32 instructions.
- PyTorch efficient FP32 uses its installed FastF32 specialization and is labeled separately.

- Final accounting exposed the initial allocation race. A second short A100 attempt (2414266) was tried; it remained pending with estimated start 21:07 local at approximately 19:11, so it was canceled using the pending-state filter. No A100 measurements were collected.
