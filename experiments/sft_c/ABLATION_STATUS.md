# Med-Qwen 3.5-4B ablation: status

## Question
Does domain-adaptive pretraining (DAPT) help once SFT and RLVR are applied?

## Pipeline
Base -> (DAPT) -> SFT -> RLVR (GRPO, plain TRL + vLLM colocate). Unsloth is used for SFT only; its GRPO corrupted gradients.

## Done
- SFT A + RLVR with DAPT, and a no-DAPT proxy (SFT A applied to the plain base, then RLVR): no-DAPT >= DAPT within the confidence intervals.
- The proxy is not a clean ablation, because SFT A was trained on top of DAPT-merged weights.
- SFT C (clean arm): SFT-B-style data (191,354 train / 2,500 held-out, 30-gram contamination filter, PubMedQA yes/no capped) trained on the plain base.
  - LoRA r=128 / alpha=256, lr 2e-5, effective batch 64, 2700 steps, 7.3 s/step, 5 h 28 min, peak 23.7 GB.
  - Eval loss 1.3084 -> 0.9026 (perplexity 3.70 -> 2.47). No overfitting; gradient norm stable at 0.33-0.55.
  - Adapter remapped (`language_model.` stripped) and verified: 128/128 lora_B tensors non-zero when loaded via AutoModelForCausalLM.
- Training stall cause: Unsloth torch.compile of the training graph. Fix: `UNSLOTH_COMPILE_DISABLE=1` before `import unsloth`, `GRAD_CKPT=False`.

## Next
1. Evaluate `SFT-C-noDAPT` in a fresh runtime (`eval_vllm_models.py`, `ONLY = ["SFT-C-noDAPT"]`).
2. RLVR from SFT C: `plain_trl_cell1_setup_sftc.py`, then cell 2 with a NEW output folder and the same settings as earlier runs.
3. Matched DAPT arm: rerun the SFT C notebook with `USE_DAPT=True` on the same cache, then RLVR.
4. Evaluate all arms; report paired tests and CIs.
5. Back up the data cache (`sft_c_cache`) to Drive.

## Open items
- Seeds (single run per arm so far), PubMedQA overlap audit, reward and num_generations ablations, held-out benchmarks.
- Model card: rewrite around the ablation; fill the repo id and verify the license.
