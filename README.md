# Med-Qwen-3.5-4B-it

A 4B medical question-answering model built on **Qwen3.5-4B-Base** in three stages:
domain-adaptive pretraining (DAPT), supervised fine-tuning (SFT), and reinforcement learning with verifiable
rewards (RLVR, GRPO). This repository holds the code, the evaluation harness and the full results.

**Weights on Hugging Face:** [`adi274903/Med-Qwen-3.5-4B-it`](https://huggingface.co/adi274903/Med-Qwen-3.5-4B-it) (merged model, ready to use).

The model writes a short reasoning trace inside `<think> ... </think>` and then gives a final answer.
After RLVR the reasoning is much shorter (about 60 tokens on average, against about 450 for the SFT model) at
the same accuracy on the main multiple-choice exams, and it is clearly better on PubMedQA.

> **Not a medical device.** This is a research model. It can be wrong and must not be used for diagnosis,
> treatment decisions, or any clinical purpose without expert review.

## Model lineage

| Stage | What it is | Artifact |
|---|---|---|
| Base | `Qwen/Qwen3.5-4B-Base` | n/a |
| 1. DAPT | Continued pretraining on biomedical text (LoRA) | `adi274903/Med-Qwen-3.5-4B-Base-DAPT-Adapter` |
| 2. SFT | Instruction / reasoning fine-tuning on top of merged DAPT (LoRA) | `adi274903/Med-Qwen-3.5-4B-SFT-Adapter` |
| 3. RLVR | GRPO with rule-based rewards on top of merged DAPT + SFT (LoRA) | merged into the released model |
| Released | Base + DAPT + SFT + RLVR, all merged into one model | `adi274903/Med-Qwen-3.5-4B-it` |

Each adapter was trained on the *merged* previous stage, so they have to be merged in order.

## Training (RLVR stage)

- **Algorithm:** GRPO (TRL `GRPOTrainer`) with vLLM rollouts, plain TRL + PEFT, 300 steps.
- **Adapter:** LoRA r=64, alpha=128 on all attention and MLP projections (about 85M trainable parameters, 2% of the model).
- **Prompts:** about 226k prompts with automatically checkable answers: medical multiple-choice, PubMedQA-style yes/no/maybe, and grade-school math.
- **Rewards (rule-based, no learned reward model):**
  - *Alignment / format:* the response has exactly one well-formed `<think>...</think>` block followed by a final answer.
  - *Accuracy:* +2.0 for a correct final answer, -1.5 for a wrong one.
  - *Efficiency:* a small bonus for shorter reasoning.
- **Optimizer / schedule:** learning rate 2e-5 with decay, no KL penalty, completions masked when truncated.
- **Stability:** gradient norm stayed at about 0.2 to 1.8 for the whole run, with no NaNs or spikes.

## Evaluation

Everything below was produced with one evaluation harness, applied identically to every stage
(vLLM, greedy decoding, 95% Wilson confidence intervals). Numbers are **accuracy %** with the interval in brackets.

**Sets:** MedQA (USMLE, 4 options): 1,273 test questions. PubMedQA: first 500 of the `pqa_labeled` set.
MedMCQA: first 1,000 validation questions. GPQA Diamond: 198 questions (options shuffled with a fixed seed).
GSM8K: first 150 test questions, 8-shot.

### Generative evaluation (chat template, model reasons then answers)

This is the main comparison: it is the setting the model was trained for.

| Benchmark | SFT | RLVR | RLVR minus SFT |
|---|---|---|---|
| MedQA (USMLE) | 73.2 [70.7-75.6] | 73.5 [71.0-75.9] | +0.3 |
| PubMedQA | 61.8 [57.5-66.0] | **75.2 [71.2-78.8]** | **+13.4** |
| MedMCQA | 62.4 [59.4-65.3] | 60.3 [57.2-63.3] | -2.1 |
| GPQA Diamond | 48.5 [41.6-55.4] | 45.5 [38.7-52.4] | -3.0 |

Output length and format validity:

| Benchmark | SFT: format valid / mean tokens | RLVR: format valid / mean tokens |
|---|---|---|
| MedQA (USMLE) | 99.4% / 456 | 100% / 66 |
| PubMedQA | 100% / 419 | 100% / 59 |
| MedMCQA | 99.6% / 359 | 100% / 65 |
| GPQA Diamond | 91.4% / 650 | 99.0% / 112 |

### Log-likelihood evaluation across all stages (no reasoning: answer scored directly)

| Benchmark | Base | + DAPT | + SFT | + RLVR |
|---|---|---|---|---|
| MedQA (USMLE) | 67.5 [64.9-70.0] | 67.0 [64.4-69.5] | 69.0 [66.4-71.5] | 69.6 [67.0-72.1] |
| PubMedQA | 43.4 [39.1-47.8] | 49.4 [45.0-53.8] | 56.4 [52.0-60.7] | 71.4 [67.3-75.2] |
| MedMCQA | 58.6 [55.5-61.6] | 58.5 [55.4-61.5] | 59.9 [56.8-62.9] | 60.1 [57.0-63.1] |
| GPQA Diamond | 43.4 [36.7-50.4] | 40.4 [33.8-47.4] | 44.9 [38.2-51.9] | 43.4 [36.7-50.4] |
| GSM8K (8-shot) | 84.0 [77.3-89.0] | 69.3 [61.5-76.2] | 88.7 [82.6-92.8] | 84.7 [78.0-89.6] |

### How to read these results

- **Efficiency is the clearest gain.** RLVR answers MedQA with the same accuracy using about 7x fewer generated tokens (66 vs 456).
- **PubMedQA improves a lot** (+13.4 points generative, +15 points log-likelihood; intervals do not overlap). The log-likelihood
  gain shows that the model changed how it answers yes/no/maybe directly, not only how it reasons. See the caveat below.
- **MedQA, MedMCQA and GPQA do not change significantly.** The differences (+0.3, -2.1, -3.0) are within the confidence
  intervals. RLVR did not add medical knowledge; it changed the format and length of the reasoning.
- **GPQA Diamond trends slightly down** (-3.0, not significant with n=198). This is the benchmark where long reasoning should matter most,
  and the RLVR model reasons for only about 112 tokens there. Treat it as a possible cost of the shortened reasoning.
- **Domain pretraining (DAPT) alone hurt math** (GSM8K 84.0 to 69.3); SFT recovered it (88.7). DAPT's only visible gain on its own
  was PubMedQA. Whether DAPT is worth keeping is tested in the [DAPT ablation](#dapt-ablation) below: after RLVR, removing it
  did not lower any score beyond the confidence intervals.

## DAPT ablation

Does the DAPT stage matter? The same SFT and RLVR recipes were applied with and without it.

| Arm | Stages |
|---|---|
| SFT / RLVR | Base + DAPT + SFT A (+ RLVR) |
| SFT-noDAPT / RLVR-noDAPT | Base + SFT A (+ RLVR); DAPT skipped |
| SFT-C-noDAPT / RLVR-C-noDAPT | Base + SFT C (+ RLVR): a different SFT dataset trained directly on the plain base, so a clean no-DAPT pipeline (but a different SFT mix, see below) |

> **Proxy caveat.** SFT A was trained on top of the DAPT-merged weights. The "noDAPT" arms apply that same adapter to the plain base,
> so they are a mismatched proxy, not a clean ablation. SFT-C is trained on the plain base, but on a different SFT dataset (with a
> benchmark-overlap filter), so it changes two things at once. A matched DAPT arm on the SFT C data was not run.

RLVR rows use the second evaluation pass of the same adapter, which is the pass the noDAPT arms were evaluated in.

### Generative evaluation (chat template, accuracy % [95% CI])

| Benchmark | SFT | SFT-noDAPT | RLVR | RLVR-noDAPT |
|---|---|---|---|---|
| MedQA (USMLE) | 73.2 [70.7-75.6] | 75.6 [73.1-77.9] | 73.6 [71.1-76.0] | 74.3 [71.8-76.6] |
| PubMedQA | 61.8 [57.5-66.0] | 66.2 [61.9-70.2] | 75.4 [71.4-79.0] | 77.2 [73.3-80.7] |
| MedMCQA | 62.4 [59.4-65.3] | 63.5 [60.5-66.4] | 60.5 [57.4-63.5] | 61.9 [58.8-64.9] |
| GPQA Diamond | 48.5 [41.6-55.4] | 38.9 [32.4-45.8] | 41.9 [35.3-48.9] | 46.5 [39.7-53.4] |
| MMLU Med (n=810) | n/a | 83.8 [81.1-86.2] | 81.5 [78.7-84.0] | 82.5 [79.7-84.9] |

Mean generated tokens / format-valid:

| Benchmark | RLVR | RLVR-noDAPT |
|---|---|---|
| MedQA (USMLE) | 66 / 100% | 96 / 100% |
| PubMedQA | 59 / 100% | 110 / 100% |
| MedMCQA | 65 / 100% | 74 / 100% |
| MMLU Med | 66 / 99.9% | 78 / 100% |
| GPQA Diamond | 108 / 99.5% | 308 / 94.9% |

### Log-likelihood evaluation (no reasoning, accuracy % [95% CI])

| Benchmark | Base | SFT | SFT-noDAPT | RLVR | RLVR-noDAPT |
|---|---|---|---|---|---|
| MedQA (USMLE) | 67.5 [64.9-70.0] | 69.0 [66.4-71.5] | 70.3 [67.7-72.8] | 69.6 [67.0-72.1] | 68.8 [66.2-71.3] |
| PubMedQA | 43.4 [39.1-47.8] | 56.4 [52.0-60.7] | 46.2 [41.9-50.6] | 71.4 [67.3-75.2] | 72.4 [68.3-76.1] |
| MedMCQA | 58.6 [55.5-61.6] | 59.9 [56.8-62.9] | 59.9 [56.8-62.9] | 60.1 [57.0-63.1] | 61.0 [57.9-64.0] |
| GPQA Diamond | 43.4 [36.7-50.4] | 44.9 [38.2-51.9] | 39.9 [33.3-46.8] | 43.4 [36.7-50.4] | 39.9 [33.3-46.8] |
| MMLU Med | n/a | n/a | 83.1 [80.3-85.5] | 80.1 [77.2-82.7] | 82.3 [79.6-84.8] |
| GSM8K (8-shot) | 84.0 [77.3-89.0] | 88.7 [82.6-92.8] | 86.7 [80.3-91.2] | 84.7 [78.0-89.6] | 80.0 [72.9-85.6] |

### SFT-C arms (clean no-DAPT pipeline, different SFT data)

Generative evaluation, accuracy % [95% CI]; "fmt" is the share of replies with a parseable `<think> ... </think>` answer.

| Benchmark | RLVR-noDAPT | SFT-C-noDAPT | RLVR-C-noDAPT |
|---|---|---|---|
| MedQA (USMLE) | 74.3 [71.8-76.6] | 56.3 [53.6-59.0] (fmt 74%) | 70.9 [68.4-73.4] (fmt 100%) |
| PubMedQA | 77.2 [73.3-80.7] | 69.4 [65.2-73.3] (fmt 90%) | 45.6 [41.3-50.0] (fmt 62%) |
| MedMCQA | 61.9 [58.8-64.9] | 53.0 [49.9-56.1] (fmt 86%) | 61.7 [58.6-64.7] (fmt 100%) |
| GPQA Diamond | 46.5 [39.7-53.4] | 32.3 [26.2-39.1] (fmt 71%) | 40.9 [34.3-47.9] (fmt 100%) |
| MMLU Med | 82.5 [79.7-84.9] | 70.9 [67.6-73.9] (fmt 85%) | 80.9 [78.0-83.4] (fmt 100%) |

Mean generated tokens for RLVR-C-noDAPT: 59 to 71 on every benchmark (62 MedQA, 59 PubMedQA, 60 MedMCQA, 59 MMLU Med, 71 GPQA).

- **SFT-C scores are depressed by format failures.** Its replies are parseable only 71 to 90% of the time (only 39% of its SFT rows contain reasoning).
  Among parseable replies its accuracy is close to SFT-noDAPT. RLVR repairs the format to 100% on four of five benchmarks.
- **PubMedQA is the exception.** RLVR-C-noDAPT has only 62% parseable replies on PubMedQA (about 73% accuracy among them). The cause has not been identified,
  so the 45.6% should not be read as a knowledge gap.
- **Elsewhere RLVR-C-noDAPT is within a few points of the other RLVR models** (MedQA 70.9 vs 73.6 and 74.3, MedMCQA 61.7, MMLU Med 80.9, GPQA 40.9) with
  overlapping intervals. The SFT C data did not improve on SFT A.
- **The earlier RLVR prompt pool was not contaminated** (see Limitations), so this gap is not explained by decontamination.

### What the ablation shows

- **After RLVR, DAPT shows no measurable benefit.** Every generative difference between RLVR and RLVR-noDAPT (+0.7, +1.8, +1.4, +4.6, +1.0 points,
  in favour of no DAPT) lies within the overlapping confidence intervals, and the no-DAPT model is nominally ahead on all five benchmarks.
- **Before RLVR, DAPT's one visible gain is PubMedQA by log-likelihood** (SFT 56.4 vs SFT-noDAPT 46.2, intervals do not overlap). RLVR then removes the gap
  (71.4 vs 72.4), so RLVR on PubMedQA-style prompts supplies what DAPT had given.
- **SFT-noDAPT is not worse generatively** (MedQA 75.6 vs 73.2, PubMedQA 66.2 vs 61.8, MedMCQA 63.5 vs 62.4; all within the intervals). Its GPQA score (38.9 vs 48.5, also within
  the intervals) comes with only 80.8% valid replies at 981 mean tokens, so part of that drop is truncation.
- **RLVR-noDAPT reasons longer** (66 to 110 tokens on the exam sets and 308 on GPQA, against 59 to 108 with DAPT). The accuracy is similar, but the efficiency gain is smaller.
- **DAPT cost math before SFT** (GSM8K 84.0 to 69.3, see above). After RLVR, GSM8K is 84.7 with DAPT and 80.0 without it, a gap inside the intervals.
- **The caveats matter.** One seed per arm, adapters that were mismatched in the noDAPT arms, and benchmark subsets all limit how far this goes.
  Read it as "no evidence that DAPT helps here", not as proof that it does not.

## Comparison with other ~4B models

The same harness was run on other recent small models. All numbers are accuracy % with 95% Wilson intervals, greedy decoding, each model
using its own chat template. **Read the format-valid column before comparing accuracy** (see the caveats below).

### Generative evaluation

| Benchmark | SFT (ours) | **RLVR (ours)** | MedGemma-1.5-4B | Gemma-4-E4B | Ministral-3-3B-Reasoning | Qwen3.5-4B | Qwen3-4B-Thinking-2507 |
|---|---|---|---|---|---|---|---|
| MedQA (USMLE) | 73.2 [70.7-75.6] | 73.5 [71.0-75.9] | 54.9 [52.2-57.6] | 75.6 [73.1-77.9] | 56.5 [53.7-59.2] | 68.3 [65.7-70.8] | 70.8 [68.2-73.2] |
| PubMedQA | 61.8 [57.5-66.0] | 75.2 [71.2-78.8] | 63.4 [59.1-67.5] | 71.4 [67.3-75.2] | 68.4 [64.2-72.3] | 66.8 [62.6-70.8] | 75.0 [71.0-78.6] |
| MedMCQA | 62.4 [59.4-65.3] | 60.3 [57.2-63.3] | 54.4 [51.3-57.5] | 63.4 [60.4-66.3] | 51.1 [48.0-54.2] | 54.4 [51.3-57.5] | 63.0 [60.0-65.9] |
| GPQA Diamond | 48.5 [41.6-55.4] | 45.5 [38.7-52.4] | 11.1 [7.5-16.2] | 41.4 [34.8-48.4] | 30.8 [24.8-37.6] | 24.2 [18.8-30.7] | 25.3 [19.7-31.7] |
| MMLU Med (4 subjects, n=810) | not run | 81.5 [78.7-84.0] (rerun) | 66.9 [63.6-70.1] | 84.8 [82.2-87.1] | 74.6 [71.5-77.4] | 77.7 [74.7-80.4] | 84.7 [82.0-87.0] |

### Answer length and format validity (generative evaluation)

"Format valid" is the share of replies from which a final answer could be extracted. For the thinking models, an invalid reply is almost always
one that ran out of tokens before answering, so low values mean the accuracy above is an **underestimate caused by the token limit**.

| Model | Max new tokens | MedQA tokens / valid | PubMedQA tokens / valid | MedMCQA tokens / valid | MMLU Med tokens / valid | GPQA tokens / valid |
|---|---|---|---|---|---|---|
| SFT (ours) | 1,536 | 456 / 99% | 419 / 100% | 359 / 100% | n/a | 650 / 91% |
| **RLVR (ours)** | 1,536 | **66 / 100%** | **59 / 100%** | **65 / 100%** | **66 / 100%** (rerun) | **112 / 99%** |
| MedGemma-1.5-4B | 1,536 | 1,259 / 82% | 745 / 99% | 1,049 / 95% | 1,084 / 93% | 1,509 / 39% |
| Gemma-4-E4B | 1,536 | 748 / 99% | 401 / 100% | 558 / 100% | 560 / 100% | 1,303 / 66% |
| Ministral-3-3B-Reasoning | 6,000 | 674 / 98% | 295 / 100% | 374 / 99% | 434 / 99% | 2,003 / 85% |
| Qwen3.5-4B | 6,000 | 4,102 / 75% | 3,133 / 82% | 3,791 / 70% | 2,997 / 83% | 5,609 / 31% |
| Qwen3-4B-Thinking-2507 | 6,000 | 3,792 / 91% | 2,555 / 100% | 3,108 / 96% | 2,725 / 97% | 5,450 / 34% |

### Log-likelihood evaluation (answer scored without reasoning)

Instruction-tuned non-Qwen models (MedGemma, Gemma) are scored inside their chat template; the Qwen-family models (including ours) use raw-text scoring,
so compare these rows within a family and treat cross-family differences with caution.

| Benchmark | Base | DAPT | SFT | RLVR | MedGemma-1.5-4B | Gemma-4-E4B | Qwen3.5-4B | Qwen3-4B-Thinking-2507 |
|---|---|---|---|---|---|---|---|---|
| MedQA (USMLE) | 67.5 | 67.0 | 69.0 | 69.6 | 44.9 | 60.7 | 71.0 | 57.7 |
| PubMedQA | 43.4 | 49.4 | 56.4 | 71.4 | 35.4 | 68.2 | 68.8 | 60.8 |
| MedMCQA | 58.6 | 58.5 | 59.9 | 60.1 | 38.3 | 54.2 | 62.0 | 54.4 |
| GPQA Diamond | 43.4 | 40.4 | 44.9 | 43.4 | 28.3 | 29.3 | 42.9 | 33.8 |
| MMLU Med | n/a | n/a | n/a | 80.1 (rerun) | 53.6 | 77.2 | 81.7 | 74.7 |
| GSM8K (8-shot) | 84.0 | 69.3 | 88.7 | 84.7 | 61.3 | 80.7 | 85.3 | 93.3 |

### What the comparison shows

- **Accuracy per token is where RLVR stands out.** On MedQA, the RLVR model (73.5%) is statistically tied with the strongest small models in this
  set (Gemma-4-E4B 75.6%, Qwen3-4B-Thinking 70.8%) while generating 66 tokens on average. The thinking models need 750 to 4,100 tokens for similar accuracy.
- **PubMedQA:** RLVR (75.2%) is tied with Qwen3-4B-Thinking (75.0%) and above the other models, subject to the training-overlap caveat below.
- **It does not win everywhere.** Gemma-4-E4B is as good or better on MedQA, MedMCQA and MMLU Med, and Qwen3-4B-Thinking reaches 84.7% on MMLU Med.
  The MedMCQA differences between RLVR (60.3%), Gemma-4-E4B (63.4%) and Qwen3-4B-Thinking (63.0%) are within the confidence intervals.
- **GPQA Diamond is not a reliable comparison here.** Several models fail the format check on most GPQA questions (Qwen3.5-4B 31%, Qwen3-4B-Thinking 34%,
  MedGemma 39%, Gemma-4-E4B 66%) because they run out of tokens, so their low GPQA scores mostly measure the token limit. Only our models and Ministral
  completed most GPQA answers, and RLVR's 45.5% is not significantly different from SFT's 48.5% or Gemma-4-E4B's 41.4%.
- **MedGemma-1.5-4B scores lower here than in Google's own reports** (for example about 69% MedQA published against 54.9% in our generative run). Our
  prompt asks for step-by-step reasoning with a fixed answer format, which is not the prompt Google used, and 18% of its MedQA replies had no extractable
  answer. Treat our MedGemma numbers as "MedGemma under this harness," not as its best result.
- **Larger models** are compared in the next section.

## Comparison with larger (~26 to 27B) models

Two larger open models were run through the same harness (MoE and dense, much more memory than our 4B model). RLVR numbers here are from a second evaluation pass of
the same adapter, so a few differ slightly from the tables above (see the run-to-run note under Limitations).

- **Qwen3.8-27B-FP8**: dense 27B reasoning model (FP8 weights), up to 6,000 new tokens.
- **Gemma-4-26B-A4B-it**: mixture-of-experts, about 26B total and about 4B active parameters per token, no thinking mode, up to 1,536 new tokens.

### Generative evaluation

| Benchmark | SFT (ours) | **RLVR (ours)** | Qwen3.8-27B-FP8 | Gemma-4-26B-A4B |
|---|---|---|---|---|
| MedQA (USMLE) | 73.2 [70.7-75.6] | 73.6 [71.1-76.0] | 86.1 [84.1-87.9] | 91.4 [89.7-92.8] |
| PubMedQA | 61.8 [57.5-66.0] | 75.4 [71.4-79.0] | 71.0 [66.9-74.8] | 73.2 [69.2-76.9] |
| MedMCQA | 62.4 [59.4-65.3] | 60.5 [57.4-63.5] | 70.5 [67.6-73.2] | 76.1 [73.4-78.6] |
| GPQA Diamond | 48.5 [41.6-55.4] | 41.9 [35.3-48.9] | 45.5 [38.7-52.4] | 70.2 [63.5-76.1] |
| MMLU Med (n=810) | not run | 81.5 [78.7-84.0] | 90.1 [87.9-92.0] | 91.1 [89.0-92.9] |

| Model | MedQA tokens / valid | PubMedQA tokens / valid | MedMCQA tokens / valid | MMLU Med tokens / valid | GPQA tokens / valid |
|---|---|---|---|---|---|
| **RLVR (ours)** | **66 / 100%** | **59 / 100%** | **65 / 100%** | **66 / 100%** | **108 / 99.5%** |
| Qwen3.8-27B-FP8 | 1,760 / 88% | 1,163 / 96% | 1,776 / 87% | 1,056 / 94% | 4,182 / 48% |
| Gemma-4-26B-A4B | 534 / 99.8% | 294 / 100% | 451 / 99.8% | 442 / 99.8% | 936 / 93% |

### Log-likelihood evaluation (no reasoning)

| Benchmark | SFT (ours) | **RLVR (ours)** | Qwen3.8-27B-FP8 | Gemma-4-26B-A4B |
|---|---|---|---|---|
| MedQA (USMLE) | 69.0 [66.4-71.5] | 69.6 [67.0-72.1] | 82.8 [80.6-84.8] | 80.8 [78.5-82.8] |
| PubMedQA | 56.4 [52.0-60.7] | 71.4 [67.3-75.2] | 57.0 [52.6-61.3] | 60.8 [56.5-65.0] |
| MedMCQA | 59.9 [56.8-62.9] | 60.1 [57.0-63.1] | 68.2 [65.2-71.0] | 66.4 [63.4-69.3] |
| GPQA Diamond | 44.9 [38.2-51.9] | 43.4 [36.7-50.4] | 49.5 [42.6-56.4] | 48.5 [41.6-55.4] |
| MMLU Med | n/a | 80.1 [77.2-82.7] | 90.6 [88.4-92.4] | 87.9 [85.5-90.0] |
| GSM8K (8-shot) | 88.7 [82.6-92.8] | 84.7 [78.0-89.6] | 97.3 [93.3-99.0] | 69.3 [61.5-76.2] |

(Gemma-4-26B-A4B is scored in its chat template; Qwen models and ours use raw-text scoring. Its low GSM8K score comes from the raw 8-shot format and is probably a
protocol effect, not a measure of its math ability.)

### Gap to RLVR (generative, points; `*` = 95% intervals do not overlap)

| Benchmark | Qwen3.8-27B-FP8 | Gemma-4-26B-A4B |
|---|---|---|
| MedQA (USMLE) | +12.5 * | +17.8 * |
| PubMedQA | -4.4 | -2.2 |
| MedMCQA | +10.0 * | +15.6 * |
| GPQA Diamond | +3.5 | +28.3 * |
| MMLU Med | +8.6 * | +9.6 * |

### What this shows

- **Larger models are clearly better on knowledge benchmarks.** On MedQA, MedMCQA and MMLU Med the two larger models lead the 4B RLVR model by roughly 9 to 18 points,
  with non-overlapping intervals. A 4B model trained with this pipeline does not close that gap.
- **PubMedQA is the exception.** RLVR (75.4%) is statistically tied with both larger models (71.0% and 73.2%), subject to the training-overlap caveat above.
- **Reasoning length differs by one to two orders of magnitude.** RLVR answers in about 60 to 110 tokens; Gemma-4-26B-A4B uses 300 to 940 and Qwen3.8-27B uses 1,000 to 4,200.
  Gemma-4-26B-A4B only activates about 4B parameters per token, so its per-token compute is comparable to a 4B model, but it needs the memory of a 26B model.
- **Qwen3.8-27B's accuracy is understated.** Its format-valid rate is 87 to 88% on MedQA and MedMCQA and only 48% on GPQA, so some correct answers were lost to the 6,000-token limit.
  The GPQA gap between it and RLVR (+3.5, not significant) is not meaningful for that reason.
- **GPQA Diamond:** Gemma-4-26B-A4B is far ahead (70.2% vs 41.9%) with 93% valid answers; that gap is real under this harness.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "adi274903/Med-Qwen-3.5-4B-it"
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, device_map="auto")

question = (
    "Question: A 54-year-old man presents with ...\n"
    "A. ...\nB. ...\nC. ...\nD. ...\n\n"
    'Think step by step, then finish with "The correct answer is X." where X is A, B, C, or D.'
)
prompt = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
if not prompt.rstrip().endswith("<think>"):
    prompt += "<think>\n"                            # the model was trained to start inside a think block

inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

The output has the form `reasoning ... </think> final answer`. For serving, vLLM loads the merged model directly.

To rebuild the merged checkpoint from the stage adapters, see [`release/ship_release.py`](release/ship_release.py).

## Repository layout and how to reproduce

```
pipeline/
  sft_a/    01_load_model.py  02_build_sft_data.py  03_train_sft.py     SFT stage (Unsloth), on top of merged DAPT
  rlvr/     01_merge_dapt_sft.py  02_build_prompt_pool_v3.py  03_train_grpo_vllm.py     RLVR stage (plain TRL + vLLM colocate)
data_audit/ rlvr_alignment_data_v4.py (decontaminated prompt pool)  audit_rlvr_contamination.py (overlap audit)
eval/       eval_vllm_models.py (all models, one harness)  eval_vllm_vs_27b.py (26-27B comparison)
release/    ship_release.py (merge, smoke test, upload)  build_model_card.py  MODEL_CARD.md
experiments/sft_c/   the clean no-DAPT SFT C arm used in the ablation (notebook, data builder, RLVR cells, format diagnostic)
```

Run order for the released model (each file is a Colab cell; paths point at Google Drive): `pipeline/sft_a/01..03` (SFT on top of the merged DAPT adapter) →
`pipeline/rlvr/01..03` (merge, build the prompt pool, GRPO) → `eval/eval_vllm_models.py` (in a fresh runtime, vLLM cannot share a process with training) → `release/ship_release.py`.

**Not in this repository:** the DAPT (continued pretraining) code and corpus build. The DAPT adapter is on the Hub
([`adi274903/Med-Qwen-3.5-4B-Base-DAPT-Adapter`](https://huggingface.co/adi274903/Med-Qwen-3.5-4B-Base-DAPT-Adapter)) and the SFT adapter is
[`adi274903/Med-Qwen-3.5-4B-SFT-Adapter`](https://huggingface.co/adi274903/Med-Qwen-3.5-4B-SFT-Adapter); the merge script pulls both from there. Training logs and checkpoints are not included either.
The released RLVR run used the prompt pool built by `pipeline/rlvr/02_build_prompt_pool_v3.py`; the decontaminated pool in `data_audit/` was used only for the later SFT C comparison.

## Benchmark overlap (contamination)

**Method.** A prompt or training row counts as overlapping if it contains a 30-word window (lower-cased, alphanumerics only) that also occurs in a
benchmark test question, or if one of its lines equals a test question exactly. The benchmark index covers the MedQA test set (1,273), MedMCQA validation (4,183),
PubMedQA-labeled (1,000, question and abstract), four MMLU medical subjects (810), GPQA Diamond (198) and GSM8K test (1,319). This finds verbatim
and near-verbatim copies. It does not find paraphrased or reformatted copies.

**Measured**

| Data | Overlapping | Detail |
|---|---|---|
| RLVR prompt pool (used for the released RLVR adapter), rebuilt from the same sources for the audit | **50 of 192,248 unique prompts (0.03%)** | multiple-choice 47 of 51,285 (0.09%), PubMedQA-style 0 of 19,999, math 0 of 27,441, general 3 of 93,523. By source: MedQA-train 46 of 10,178 (0.45%), medical-o1 3 of 19,704, MedMCQA-train 1 of 39,989 |
| SFT-stage sources (`pipeline/sft_a`), checked source by source | about **47 rows, about 0.02% of the mix** | `medical_meadow_medqa` (10,178 rows, the same size as the MedQA train split): **44 rows (0.43%)** overlap a MedQA test question, the same count the MedQA train split itself shows under this filter, so this is MedQA's own near-duplication between its train and test splits and no full test question appears in it. medical-o1-reasoning-SFT: 3 of 19,704 rows, and II-Medical, AlpaCare, UltraChat and Orca-Math: none flagged, from the earlier per-source audit run with slightly different row caps. The exact SFT cache was not rebuilt for this audit |
| SFT C training data | 50 rows removed before training | the same sources and the same split: 46 MedQA-train, 3 medical-o1, 1 MedMCQA-train. The SFT C and RLVR-C models were trained without them |

The RLVR-C-noDAPT run used a prompt pool with those 50 prompts filtered out; the released DAPT and no-DAPT RLVR runs used the unfiltered pool. Since the pool
is 99.97% clean, the difference between the runs is not explained by contamination.

**Not measured**
- The exact SFT A cache was not rebuilt, so its overlap is estimated from its sources (see the table), not counted on the cache itself. The sources, caps and filters match closely but not exactly.
- The DAPT corpus was not checked.
- Paraphrased overlap was not checked for any data.

**By-design overlap.** The training sets include the train splits of MedQA, MedMCQA and PubMedQA (the artificial split), and GSM8K and ARC. The
benchmarks above are therefore in-distribution for this model. That is standard practice, but it means the scores are not zero-shot. The held-out test splits were not used for training.

## Limitations

- **Harness-specific numbers.** Results come from our own evaluation harness with a single shared prompt and parser, on subsets of some
  benchmarks (PubMedQA 500, MedMCQA 1,000, GSM8K 150). They are not directly comparable to vendor-reported figures that use different
  prompts, shots, or splits.
- **Uneven token budgets in the cross-model comparison.** Our models and the Gemma/MedGemma models were limited to 1,536 new tokens; the thinking models
  (Qwen3.5-4B, Qwen3-4B-Thinking, Ministral) got 6,000. Models with low format-valid rates are undercounted. A fairer comparison would raise the budget until
  every model's format-valid rate is above about 95%.
- **MMLU Med was only run for the final RLVR model** (in a second pass), so the SFT/DAPT/Base entries for that benchmark are empty.
- **Run-to-run variation.** The RLVR model was evaluated twice with the same settings. Most numbers moved by 0.1 to 0.2 points, but GPQA Diamond
  moved from 45.5% to 41.9% (7 questions out of 198), probably from non-deterministic batching in vLLM. Differences of a few points on small benchmarks
  are within this noise.
- **Contamination was audited only for verbatim matches, and not for all data.** See the Benchmark overlap section: the RLVR prompt pool is 99.97% clean at 30 words and the SFT sources show a similar rate, but the DAPT corpus was not audited and the exact SFT cache was estimated from its sources.
- **Short reasoning.** The efficiency reward drove reasoning down to roughly the minimum allowed length. This is fine on exam-style
  questions but may be a weakness on problems that need long, multi-step reasoning (see GPQA).
- **No safety tuning for clinical use.** The model can state incorrect facts confidently and was not evaluated for bias, calibration,
  or harmful-advice behaviour.
- **Single training run, single seed.** Differences of a few points between stages are within noise.
- **The no-DAPT ablation is a proxy.** SFT A was trained on DAPT-merged weights and applied to the plain base for the noDAPT arms. SFT C is a clean no-DAPT pipeline but uses a different SFT dataset, and a matched DAPT arm on that dataset was not run.
- **English only**, and text only.

## License

Apache License 2.0 (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). The model is a modified derivative of Qwen3.5-4B-Base (Apache-2.0).
Datasets and benchmarks used for training and evaluation keep their own licenses and are not redistributed here.

## Citation and references

If you use this model or code, please cite it (GitHub also offers "Cite this repository" from [`CITATION.cff`](CITATION.cff)):

```bibtex
@misc{sannabhadti2026medqwen,
  title  = {Med-Qwen-3.5-4B-it: a 4B medical question-answering model with short reasoning (DAPT, SFT, GRPO with verifiable rewards)},
  author = {Sannabhadti, Aditya},
  year   = {2026},
  url    = {https://github.com/adi274903/med-qwen-3.5-4b-it},
  note   = {Model: https://huggingface.co/adi274903/Med-Qwen-3.5-4B-it}
}
```

This work builds on Qwen3.5-4B-Base (Qwen Team) and on public datasets and benchmarks (among them MedQA, MedMCQA, PubMedQA, MMLU, GPQA, GSM8K, medical-o1-reasoning-SFT,
II-Medical-Reasoning-SFT, AlpaCare, UltraChat and Orca-Math), and on LoRA, DoRA, GRPO and RLVR. Full references with links and where each is used are in
[`CITATIONS.md`](CITATIONS.md); BibTeX for all of them is in [`references.bib`](references.bib). Please cite the base model and any dataset or benchmark whose numbers you report.
