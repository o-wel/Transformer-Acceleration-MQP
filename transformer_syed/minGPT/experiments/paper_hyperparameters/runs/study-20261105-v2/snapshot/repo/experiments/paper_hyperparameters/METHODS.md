# Protocol for hyperparameter selection and independent confirmation

## Research question and scope

For this GPT-2 implementation, a fixed NVIDIA A30 GPU class, and a controlled synthetic basketball task, which tested LoRA, full-context KV-cache prefill, and FA1-derived attention settings provide the best quality/cost tradeoff? Do the independently selected settings reduce inference latency when combined, relative to a predeclared untuned combination?

The pilot is exploratory evidence only. Its model checkpoints, timing measurements and previously examined test scores are not pooled into this study. This protocol and all executable sources/data are hashed before the study begins. “Best tested under a declared objective” is the appropriate paper wording. Longer experiments alone cannot establish global optimality or generalization across models, tasks or hardware.

The study distinguishes LoRA adaptation hyperparameters, cache prefill scheduling parameters, and attention kernel launch parameters. Prefill chunk size does not alter retained context. Kernel tiles and warp counts do not change the attention algorithm. Training and inference have different primary costs: LoRA selection uses training-step latency under a quality constraint; cache/attention selection uses inference latency. The combined endpoint is inference latency. KV caching is not applied during training.

## Data and protection against test reuse

Training uses the existing 480 fictional matches (1,440 questions). Before running any search, the deterministic generator with seed 20261105 produces 200 new validation matches (600 questions) and 300 new test matches (900 questions). Every match yields winner, halftime and comeback questions. Matches and score quadruples are disjoint between training, validation and test; new score quadruples also exclude every original train/validation/test/counterfactual match used in the pilot.

Training/validation retain the original eight training team names. Test uses eight new team names absent from both the original training and prior pilot test team lists. Test question templates and score distributions follow the established controlled generator; this is a study of within-task compositional generalization, not a new instruction-following or real-world sports benchmark.

Each test question has a paired counterfactual that reverses the relevant final or halftime scores, giving 900 additional related questions. All six original/counterfactual questions belonging to one underlying match share a group ID. The 1,800 questions correspond to 300 independent generated test matches. Raw prompts, outputs, token IDs and references are retained. No held-out output is generated before all selections are frozen. Five new confirmation seeds never participate in hyperparameter selection.

## LoRA discovery

The Cartesian search contains 45 settings: rank in {2,4,8,16,32}, alpha/rank in {1,2,4}, and learning rate in {0.0001,0.0003,0.001}. Every candidate is independently trained with seeds {11,22,33}, giving 135 discovery runs. Base weights are frozen; adapters remain unmerged and target fused QKV, attention output and both MLP projections in every transformer block.

Each run receives 2,400 successful optimizer updates, batch size 8, padded length 128, AdamW with constant LR, weight decay zero, gradient clipping norm 1, and dropout zero. FP32 stored weights use FP16 autocast; GradScaler starts at 128. TF32 is disabled. A nonfinite/skipped update fails the task rather than silently reducing its budget. Identical sampled example schedules are used for every candidate within a seed and verified by hashes. Frozen base weights are checked exactly before and after training.

Validation answer-token NLL is measured at steps 300,600,...,2400. Each run selects the checkpoint with the smallest NLL, choosing the earliest on an exact tie. No test score selects a checkpoint. Candidate quality is the arithmetic mean of those selected NLLs across the three discovery seeds. Candidate cost is the arithmetic mean of each seed's median warm optimizer-step latency. Step timing includes forward, backward, clipping and optimizer update, with synchronization, and excludes input construction/transfer, checkpoint IO, evaluation and auditing. The first 50 updates of each process segment are excluded from the warm-step summary. Successful update counts and all measured times remain available.

Let q* be the smallest candidate mean validation NLL. The operational LoRA setting is the candidate with the lowest mean warm training-step latency among candidates with mean NLL <= q* + 0.01. Ties resolve by lower rank, then alpha, then LR. The quality-best setting and the quality/cost Pareto frontier are also retained. The untuned control is fixed at rank 4, alpha 8, LR 0.0001. It remains a control even if it falls outside the quality constraint; quality and cost must both be reported. If this control wins, the study reports that result without manufacturing a slower control.

## Cache and attention discovery

Runtime discovery uses three independent GPU allocations, one for each discovery seed, holding that seed's untuned LoRA checkpoint fixed. The cache sweep tests prefill chunk sizes {32,64,128,256,512,1024}; 1024 means full prefill for all declared workloads. The cache dynamically retains every K/V token. It uses no truncation, eviction, quantization or reduced context. The baseline chunk is 128.

The attention sweep tests query tiles {16,32,64}, key tiles {16,32,64,128}, and warp counts {4,8}, giving 24 launch configurations. Every configuration uses the same locally copied FA1-derived Triton forward algorithm. The baseline is query/key tiles 16/32 and 4 warps. This prototype is not the original optimized FA1 CUDA implementation. This study tunes the prototype only; it includes no production FlashAttention comparison.

The six discovery workloads are batch {1,4} crossed with prompt length {32,256,896}, generating exactly 64 tokens. LoRA discovery uses the same eager training backend for every candidate. Cache discovery uses eager attention; attention discovery disables caching. Thus cache and attention parameters are selected independently. Neither batch size, prompt length, output length, precision nor retained context is optimized.

Each runtime allocation performs five excluded warmup blocks and ten measured blocks per workload. Modes are shuffled within a block and share the same deterministic random-token prompt; prompts change between blocks. Selection minimizes the geometric mean of per-allocation/per-workload median synchronized wall times, weighting each workload and each allocation equally. Only settings passing every allocation's quality gates are eligible; exact timing ties follow original grid order. Test timings and answer scores never participate in this choice.

## Numerical and implementation gates

All 24 prototype configurations are tested against FP64 eager attention using FP16 Q/K/V, head width 64, and (queries,past) in {(37,0),(17,21),(1,36),(129,0),(1,895),(1,1023)}. The prespecified tolerance is atol=0.012, rtol=0.04. Compilation/resource or correctness failures are recorded and disqualify a launch configuration; the baseline must pass. Cache settings failing preflight are likewise recorded and excluded; the baseline cache setting must pass. These checks validate forward inference; this study makes no prototype-backward/training-speed claim.

Cache checks compare final-prefill and three subsequent teacher-forced logits with uncached FP16 eager logits on identical input prefixes, including uneven chunks and long contexts, at batches 1 and 4. Tolerances are atol=0.1, rtol=0.02. Greedy argmax differences are recorded. FP16 shape-dependent rounding need not produce bitwise-identical greedy sequences. Cache and prototype settings additionally require absolute validation NLL difference <=0.01 from the same weights under eager FP16 inference. The pilot's observed chunk-dependent greedy divergence motivated these criteria before this fresh study; no gate will be loosened in response to confirmation results.

Preflight also performs a real optimizer update with rank 32/alpha 128, checks the frozen base, tests eager/cache/prototype/combined paths at batch 4, prompt 896, output 128, and preserves a prototype profiler trace and baseline PTX. Expensive jobs depend on successful preflight. Software versions, GPU UUID, node, Slurm allocation and available clock/temperature/power telemetry are recorded. Source/data/selection hashes are checked at stage entry, base weights, adapter checkpoint hashes and tokenizer vocabulary are checked across jobs, and parameter hashes must remain unchanged across inference benchmarking.

## Independent confirmation

After all discovery results are complete, selection.json is written and hashed. No runtime or quality confirmation stage can change it. Selected and untuned LoRA settings are retrained from the same pretrained base with new seeds {101,202,303,404,505}, receiving the same 2,400-update budgets and validation checkpoint opportunities. Training order is shuffled within a seed. Identical settings are trained once if the selected setting equals the control.

Each confirmation seed receives three separately allocated runtime jobs. Every job measures the six original workload shapes and eight held-out shapes: batch {1,4}, prompt {128,512}, output {32,128}. All prompt-plus-output lengths fit within the 1,024-token context window; there is no context rollover or early EOS stopping in timing. FP32 stored weights, FP16 autocast, greedy decoding and last-token-only LM-head projection are identical across modes. Timing uses five warmup blocks and twenty measured paired blocks per workload, with fresh prompts and shuffled mode order. Each block has the same prompt across all modes. Backend selection, garbage collection, telemetry, host/device prompt transfer and output hashing are outside the timed interval. Compilation is warmed up. Both synchronized wall latency and CUDA-event latency are recorded, along with absolute allocated/reserved and incremental peak allocator memory.

The confirmation modes include all 2^3 combinations of untuned/tuned LoRA, cache and prototype settings with all three methods enabled. This parameter-level factorial reveals interaction and transfer of isolated choices; it is not an on/off feature factorial. Isolated LoRA, cache and attention tuning pairs are also remeasured, with the other switches fixed as in discovery. Final speedups are computed directly from measured combinations, not products of isolated speedups. All mode comparisons within an allocation use the same resident model set; absolute timings from discovery are not pooled with confirmation.

Each confirmation seed separately evaluates both LoRA models under eager uncached inference and their respective all-three configurations on all 900 fresh test and 900 counterfactual questions. Quality decoding generates at most 64 tokens for interpretation, trims at the first EOS and following User turn, and uses the same prompts/references in every condition. The primary task metric is normalized exact reference match. Primary/numeric field scores, raw outputs, and paired counterfactual behavior supplement it; field scoring alone does not establish semantic correctness. Runtime NLL gates are rechecked on both newly trained models. A confirmation gate failure is reported and blocks a positive quality-preserving claim; it never triggers test-driven retuning.

## Estimands and uncertainty

For a paired observation define z = log(latency_untuned / latency_tuned). Within each allocation/workload, average z over paired repeats; then average equally across the fixed workloads, allocations and training seeds, and exponentiate. A ratio >1 means speedup, <1 means slowdown. The prespecified primary endpoint is the combined all-tuned/all-untuned ratio over the eight held-out workload shapes. The six original shapes and all fourteen together are secondary endpoints.

Timing 95% intervals use 5,000 hierarchical paired bootstrap draws: resample five training seeds, resample three runtime allocations within each sampled seed, and resample twenty paired repeats within each allocation/workload. Workload shapes are a fixed design and retain equal weights. Twenty repetitions on one allocation are not treated as twenty independent GPU experiments. Training-step ratios use paired seed-level medians and seed bootstrap. Hardware inference is conditional on this A30 class and allocation environment, not a claim about a population of GPU architectures.

Task-quality differences use a crossed paired bootstrap over the five training seeds and 300 held-out match groups. The same sampled matches are used across sampled seeds and conditions. Original and counterfactual questions within a match remain clustered. Original exact/field accuracy differences are primary reported task differences; paired counterfactual correctness and the eager/combined comparison are available in raw artifacts and summary tables. Secondary intervals are descriptive and are not multiple-comparison-adjusted hypothesis tests. A positive primary speed claim requires its combined held-out 95% interval to exclude 1 on the speedup side and the confirmation quality gates to pass. Faster configurations with worse task quality must be presented with the tradeoff explicitly.

The factorial table reports all eight measured combinations. Log-latency contrasts use sum(sign * log geometric-mean latency)/4 for the declared LoRA/cache/attention signs; these are descriptive interaction estimates, not additional confirmatory claims.

## Reproducibility, failures and compute cost

Sources, exact generated data and protocol are snapshotted before submission; dependency versions and PyTorch build configuration are recorded during preflight; the working repository can continue to change without changing these jobs. Each stage validates the snapshot and software versions. Checkpoints include optimizer/scaler/RNG state and deterministic example schedule hashes, permitting training to resume at a completed validation boundary without silently changing the budget. Warm training steps after a restart are excluded again. Partial timing allocations are archived and restarted in full after a new Slurm allocation; samples from distinct allocations are never silently merged.

The Slurm dependency DAG is preflight -> 45 candidate tasks (three seeds per task) -> 3 runtime-discovery tasks -> frozen selection -> 5 confirmation-training tasks -> 15 runtime-confirmation + 5 quality tasks -> automatic report. The concurrency limit is three search/runtime GPU tasks, with a fourth GPU available for quality; all jobs use the authorized isp account. Any failed predecessor prevents dependent stages from releasing. Job IDs, dependencies, commands, stage failures, raw JSON, per-step timings, per-repeat timings, outputs, hashes and CSV summaries remain in the study directory. “Submitted” is distinct from “completed.”

Search/validation/model-loading/checkpoint costs are excluded from deployment latency and must be reported as separate experimental compute using Slurm accounting and retained logs. This study does not assume tuning cost is amortized. The final report is produced only after all confirmation timing and quality jobs succeed. The paper must report finite-grid boundaries, single model/task/GPU scope, fixed precision, synthetic timing inputs, unmerged LoRA, generator/template limits, scoring limitations and any failed quality gates.

## References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*: https://arxiv.org/abs/2106.09685
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*: https://arxiv.org/abs/2205.14135
- Cawley and Talbot, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*: https://jmlr.org/papers/v11/cawley10a.html

These references motivate the methods and separation of selection/evaluation; the exact search, workload design and thresholds above are choices of this study.
