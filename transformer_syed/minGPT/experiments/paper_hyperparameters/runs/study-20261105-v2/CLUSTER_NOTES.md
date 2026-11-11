# Submission and allocation notes

This is the active prototype-only study. The v1 chain was cancelled in response to the user's scope choice before discovery training began; it is excluded.

Array concurrency requests cap search/runtime jobs at three and quality at one. The academic partition currently reports QOSMaxGRESPerUser after two A30 allocations, so the effective overall concurrency is currently two GPUs. This can lengthen queue/wall time; it changes neither the search, independent replication counts, per-job budgets nor the registered estimands. The frozen methodology's concurrency wording describes requested throttles; this note records the observed cluster restriction.

Preflight completed successfully: all 24 launch configurations (144 FP64-reference comparisons), all six cache settings (72 same-prefix logit comparisons), a real rank-32 optimizer update with frozen-base audit, and all four eager/cache/prototype/combined maximum-workload paths passed. Prototype PTX and a profiler trace are retained.
