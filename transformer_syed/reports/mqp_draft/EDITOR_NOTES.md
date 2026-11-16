# Editing and verification notes

`MQP_REPORT_UPDATED.tex` preserves the pasted background and bibliography and inserts the completed Methodology section before the citations. The two smaller `.tex` files contain only the new methodology and additional bibliography entries for insertion into another Overleaf project.

## Corrections and scope

- The custom FP32 attention implementation in this checkout uses Triton and PyTorch. The prose uses those names rather than JAX.
- Owen's supplied KV-cache and Ax material is integrated and edited for consistent terminology. Request time is TTFT plus the sum of subsequent token intervals, not TTFT plus one average TBT. Pure prefill timing is distinguished from first-token timing.
- Ax settings and its default 20-trial budget come from Owen's supplied text. No matching Ax implementation or completed Ax trial record was identified in the inspected checkout. The report describes this workflow without claiming a completed 20-trial result. Owen should verify those settings against his runner before publication.
- The separate staged FP32 search is documented as a separate experiment. Its trial methodology is not attributed to Ax.
- Initial forward-only attention measurements are distinguished from the later backward-enabled implementation.
- The model's physical parameter count includes minGPT's separately stored output head. The standard GPT-2 release count and local storage count are both explained.
- The supplied background and its existing references were preserved; they were not comprehensively re-reviewed.

## Local implementation sources

Paths below are relative to the repository root:

- Model, cache and LoRA: `transformer_syed/minGPT/mingpt/model.py`.
- FP32 attention: `transformer_syed/minGPT/mingpt/tiled_attention.py` and `transformer_syed/minGPT/experiments/fp32_attention/README.md`.
- Basketball generator and scoring: `transformer_syed/minGPT/experiments/sports/data.py`, `scoring.py`, `README.md`, `METHODS.md`, and `matched/README.md`.
- SMS preparation and classification: `transformer_syed/minGPT/experiments/sms_spam/experiment.py` and `README.md`.
- Cache timing: `transformer_syed/minGPT/experiments/kv_cache/benchmark.py` and `README.md`.
- Staged search and combined measurements: `transformer_syed/minGPT/experiments/flash_attention/results/hyperparameter_search/README.md` and `METHODS.md`.

External primary references are included in ADDITIONAL_REFERENCES.tex. Model details were checked against the GPT-2 configuration, original model card and technical report. The Ax documentation was checked for multiobjective terminology; it does not verify project-specific Ax settings.

## Validation and Overleaf

Static checks passed for citation resolution, bibliography-key uniqueness, LaTeX environment nesting, and brace counts. No LaTeX compiler was available here, so compilation and visual layout have not been verified.

The original background expects `LoRA.png`, `KVCache.png`, and `FlashAttention.png`. These assets were absent from this checkout. Keep the existing image files in the Overleaf project when uploading the updated source. The commented `framework.png` figure remains commented. This update creates no replacement figures.
