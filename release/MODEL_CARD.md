---
language:
- en
license: apache-2.0
library_name: transformers
base_model: Qwen/Qwen3.5-4B-Base
tags:
- medical
- biomedical
- reasoning
- grpo
- rlvr
- qwen3.5
pipeline_tag: text-generation
---

# Med-Qwen-3.5-4B-it

A 4B medical question-answering model built on Qwen3.5-4B-Base with domain-adaptive pretraining, supervised fine-tuning and reinforcement learning with verifiable rewards (GRPO), released as one merged checkpoint. It writes a short reasoning trace inside `<think> ... </think>` and then a final answer.

## Highlights

- **Short reasoning, competitive accuracy.** MedQA (USMLE) 73.5% with 66 generated tokens on average; the other recent small models in the comparison need 750 to 4,100 tokens for similar MedQA accuracy.
- **Statistically tied with the best small models on MedQA:** Gemma-4-E4B 75.6% (748 tokens) and Qwen3-4B-Thinking 70.8% (3,792 tokens).
- **PubMedQA 75.2%,** tied with Qwen3-4B-Thinking (75.0%, about 2,500 tokens) and above the other small models.
- **Reliable format:** 99 to 100% of answers are parseable on every benchmark; the long-reasoning models lose 9 to 69% of their answers to the token limit on some sets.
- **Honest limits:** other models score higher on MedMCQA and MMLU Med, and the model is 9 to 18 points behind 26-27B models on knowledge benchmarks.
- **Benchmark overlap:** 0.03% of the RLVR prompt pool matched a benchmark question verbatim; the SFT and DAPT data were not fully audited (details below).

> **Not a medical device.** This is a research model. It can be wrong and must not be used for diagnosis, treatment decisions, or any clinical purpose without expert review.

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
    prompt += "<think>\n"            # the model was trained to start inside a think block
inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

The output has the form `reasoning ... </think> final answer`. The final answer is either a bare option letter (for example `B`) or a sentence such as
"The correct answer is B."; when scoring, read the text after `</think>` and take the last option letter, `yes`/`no`/`maybe`, or number. vLLM loads this model directly.

## Training (RLVR stage)

- **Algorithm:** GRPO (TRL `GRPOTrainer`) with vLLM rollouts, plain TRL + PEFT, 300 steps.
- **Trained parameters:** LoRA r=64, alpha=128 on all attention and MLP projections (about 85M trainable parameters, 2% of the model).
- **Prompts:** about 226k prompts with automatically checkable answers: medical multiple-choice, PubMedQA-style yes/no/maybe, and grade-school math.
- **Rewards (rule-based, no learned reward model):**
  - *Alignment / format:* the response has exactly one well-formed `<think>...</think>` block followed by a final answer.
  - *Accuracy:* +2.0 for a correct final answer, -1.5 for a wrong one.
  - *Efficiency:* a small bonus for shorter reasoning.
- **Optimizer / schedule:** learning rate 2e-5 with decay, no KL penalty, completions masked when truncated.
- **Stability:** gradient norm stayed at about 0.2 to 1.8 for the whole run, with no NaNs or spikes.

## Evaluation

Everything below was produced with one evaluation harness (vLLM, greedy decoding, 95% Wilson confidence intervals). Numbers are **accuracy %** with the interval in brackets.

**Sets:** MedQA (USMLE, 4 options): 1,273 test questions. PubMedQA: first 500 of the `pqa_labeled` set.
MedMCQA: first 1,000 validation questions. GPQA Diamond: 198 questions (options shuffled with a fixed seed).
MMLU Med: 810 questions from four medical subjects. GSM8K: first 150 test questions, 8-shot.

### Generative evaluation (chat template, the model reasons then answers)

This is the setting the model was trained for.

| Benchmark | Accuracy | Answers parseable | Mean generated tokens |
|---|---|---|---|
| MedQA (USMLE) | 73.5 [71.0-75.9] | 100% | 66 |
| PubMedQA | 75.2 [71.2-78.8] | 100% | 59 |
| MedMCQA | 60.3 [57.2-63.3] | 100% | 65 |
| GPQA Diamond | 45.5 [38.7-52.4] | 99.0% | 112 |
| MMLU Med (n=810) | 81.5 [78.7-84.0] | 99.9% | 66 |

### Log-likelihood evaluation (no reasoning: the answer is scored directly)

| Benchmark | Accuracy |
|---|---|
| MedQA (USMLE) | 69.6 [67.0-72.1] |
| PubMedQA | 71.4 [67.3-75.2] |
| MedMCQA | 60.1 [57.0-63.1] |
| GPQA Diamond | 43.4 [36.7-50.4] |
| MMLU Med | 80.1 [77.2-82.7] |
| GSM8K (8-shot) | 84.7 [78.0-89.6] |

### How to read these results

- **Short reasoning.** The model answers exam questions with about 60 generated tokens on average (112 on GPQA Diamond), where the other recent small models in the comparison below use 400 to 5,600.
- **PubMedQA is the strongest result,** 75.2% generative and 71.4% by log-likelihood. See the benchmark-overlap section for the checks on training/test overlap.
- **MedMCQA and GPQA Diamond are not strengths:** other models score higher on MedMCQA, and the GPQA Diamond interval is wide (n=198).
- **Short reasoning may cost accuracy on problems that need long multi-step reasoning;** GPQA Diamond is the closest test here and the result is not conclusive.

## Comparison with other ~4B models

The same harness was run on other recent small models. All numbers are accuracy % with 95% Wilson intervals, greedy decoding, each model
using its own chat template. **Read the format-valid column before comparing accuracy** (see the caveats below).

### Generative evaluation

| Benchmark | **Med-Qwen-3.5-4B-it (ours)** | MedGemma-1.5-4B | Gemma-4-E4B | Ministral-3-3B-Reasoning | Qwen3.5-4B | Qwen3-4B-Thinking-2507 |
|---|---|---|---|---|---|---|
| MedQA (USMLE) | 73.5 [71.0-75.9] | 54.9 [52.2-57.6] | **75.6 [73.1-77.9]** | 56.5 [53.7-59.2] | 68.3 [65.7-70.8] | 70.8 [68.2-73.2] |
| PubMedQA | **75.2 [71.2-78.8]** | 63.4 [59.1-67.5] | 71.4 [67.3-75.2] | 68.4 [64.2-72.3] | 66.8 [62.6-70.8] | 75.0 [71.0-78.6] |
| MedMCQA | 60.3 [57.2-63.3] | 54.4 [51.3-57.5] | **63.4 [60.4-66.3]** | 51.1 [48.0-54.2] | 54.4 [51.3-57.5] | 63.0 [60.0-65.9] |
| GPQA Diamond | **45.5 [38.7-52.4]** | 11.1 [7.5-16.2] | 41.4 [34.8-48.4] | 30.8 [24.8-37.6] | 24.2 [18.8-30.7] | 25.3 [19.7-31.7] |
| MMLU Med (4 subjects, n=810) | 81.5 [78.7-84.0] (rerun) | 66.9 [63.6-70.1] | **84.8 [82.2-87.1]** | 74.6 [71.5-77.4] | 77.7 [74.7-80.4] | 84.7 [82.0-87.0] |

**Bold** = highest point estimate in the row among the ~4B models; most intervals overlap, so treat near-ties as ties. On GPQA Diamond several of the other models are cut off by the token limit (see the validity table below), so that row favours models that answer briefly.

### Answer length and format validity (generative evaluation)

"Format valid" is the share of replies from which a final answer could be extracted. For the thinking models, an invalid reply is almost always
one that ran out of tokens before answering, so low values mean the accuracy above is an **underestimate caused by the token limit**.

| Model | Max new tokens | MedQA tokens / valid | PubMedQA tokens / valid | MedMCQA tokens / valid | MMLU Med tokens / valid | GPQA tokens / valid |
|---|---|---|---|---|---|---|
| **Med-Qwen-3.5-4B-it (ours)** | 1,536 | **66 / 100%** | **59 / 100%** | **65 / 100%** | **66 / 100%** (rerun) | **112 / 99%** |
| MedGemma-1.5-4B | 1,536 | 1,259 / 82% | 745 / 99% | 1,049 / 95% | 1,084 / 93% | 1,509 / 39% |
| Gemma-4-E4B | 1,536 | 748 / 99% | 401 / 100% | 558 / 100% | 560 / 100% | 1,303 / 66% |
| Ministral-3-3B-Reasoning | 6,000 | 674 / 98% | 295 / 100% | 374 / 99% | 434 / 99% | 2,003 / 85% |
| Qwen3.5-4B | 6,000 | 4,102 / 75% | 3,133 / 82% | 3,791 / 70% | 2,997 / 83% | 5,609 / 31% |
| Qwen3-4B-Thinking-2507 | 6,000 | 3,792 / 91% | 2,555 / 100% | 3,108 / 96% | 2,725 / 97% | 5,450 / 34% |

**Bold** = fewest generated tokens in the column.

### Log-likelihood evaluation (answer scored without reasoning)

Instruction-tuned non-Qwen models (MedGemma, Gemma) are scored inside their chat template; the Qwen-family models (including ours) use raw-text scoring,
so compare these rows within a family and treat cross-family differences with caution.

| Benchmark | **Med-Qwen-3.5-4B-it (ours)** | MedGemma-1.5-4B | Gemma-4-E4B | Qwen3.5-4B | Qwen3-4B-Thinking-2507 |
|---|---|---|---|---|---|
| MedQA (USMLE) | 69.6 | 44.9 | 60.7 | **71.0** | 57.7 |
| PubMedQA | **71.4** | 35.4 | 68.2 | 68.8 | 60.8 |
| MedMCQA | 60.1 | 38.3 | 54.2 | **62.0** | 54.4 |
| GPQA Diamond | **43.4** | 28.3 | 29.3 | 42.9 | 33.8 |
| MMLU Med | 80.1 (rerun) | 53.6 | 77.2 | **81.7** | 74.7 |
| GSM8K (8-shot) | 84.7 | 61.3 | 80.7 | 85.3 | **93.3** |

**Bold** = highest point estimate in the row among the ~4B models; most intervals overlap, so treat near-ties as ties.

### What the comparison shows

- **Accuracy per token is where this model stands out.** On MedQA, this model (73.5%) is statistically tied with the strongest small models in this
  set (Gemma-4-E4B 75.6%, Qwen3-4B-Thinking 70.8%) while generating 66 tokens on average. The thinking models need 750 to 4,100 tokens for similar accuracy.
- **PubMedQA:** this model (75.2%) is tied with Qwen3-4B-Thinking (75.0%) and above the other models, subject to the training-overlap caveat below.
- **It does not win everywhere.** Gemma-4-E4B is as good or better on MedQA, MedMCQA and MMLU Med, and Qwen3-4B-Thinking reaches 84.7% on MMLU Med.
  The MedMCQA differences between this model (60.3%), Gemma-4-E4B (63.4%) and Qwen3-4B-Thinking (63.0%) are within the confidence intervals.
- **GPQA Diamond is not a reliable comparison here.** Several models fail the format check on most GPQA questions (Qwen3.5-4B 31%, Qwen3-4B-Thinking 34%,
  MedGemma 39%, Gemma-4-E4B 66%) because they run out of tokens, so their low GPQA scores mostly measure the token limit. Only our models and Ministral
  completed most GPQA answers, and our 45.5% is not significantly different from Gemma-4-E4B's 41.4%.
- **MedGemma-1.5-4B scores lower here than in Google's own reports** (for example about 69% MedQA published against 54.9% in our generative run). Our
  prompt asks for step-by-step reasoning with a fixed answer format, which is not the prompt Google used, and 18% of its MedQA replies had no extractable
  answer. Treat our MedGemma numbers as "MedGemma under this harness," not as its best result.
- **Larger models** are compared in the next section.

## Comparison with larger (~26 to 27B) models

Two larger open models were run through the same harness (MoE and dense, much more memory than our 4B model). Our numbers here are from a second evaluation pass of
the same model, so a few differ slightly from the tables above (see the run-to-run note under Limitations).

- **Qwen3.8-27B-FP8**: dense 27B reasoning model (FP8 weights), up to 6,000 new tokens.
- **Gemma-4-26B-A4B-it**: mixture-of-experts, about 26B total and about 4B active parameters per token, no thinking mode, up to 1,536 new tokens.

### Generative evaluation

| Benchmark | **Med-Qwen-3.5-4B-it (ours)** | Qwen3.8-27B-FP8 | Gemma-4-26B-A4B |
|---|---|---|---|
| MedQA (USMLE) | 73.6 [71.1-76.0] | 86.1 [84.1-87.9] | 91.4 [89.7-92.8] |
| PubMedQA | 75.4 [71.4-79.0] | 71.0 [66.9-74.8] | 73.2 [69.2-76.9] |
| MedMCQA | 60.5 [57.4-63.5] | 70.5 [67.6-73.2] | 76.1 [73.4-78.6] |
| GPQA Diamond | 41.9 [35.3-48.9] | 45.5 [38.7-52.4] | 70.2 [63.5-76.1] |
| MMLU Med (n=810) | 81.5 [78.7-84.0] | 90.1 [87.9-92.0] | 91.1 [89.0-92.9] |

| Model | MedQA tokens / valid | PubMedQA tokens / valid | MedMCQA tokens / valid | MMLU Med tokens / valid | GPQA tokens / valid |
|---|---|---|---|---|---|
| **Med-Qwen-3.5-4B-it (ours)** | **66 / 100%** | **59 / 100%** | **65 / 100%** | **66 / 100%** | **108 / 99.5%** |
| Qwen3.8-27B-FP8 | 1,760 / 88% | 1,163 / 96% | 1,776 / 87% | 1,056 / 94% | 4,182 / 48% |
| Gemma-4-26B-A4B | 534 / 99.8% | 294 / 100% | 451 / 99.8% | 442 / 99.8% | 936 / 93% |

### Log-likelihood evaluation (no reasoning)

| Benchmark | **Med-Qwen-3.5-4B-it (ours)** | Qwen3.8-27B-FP8 | Gemma-4-26B-A4B |
|---|---|---|---|
| MedQA (USMLE) | 69.6 [67.0-72.1] | 82.8 [80.6-84.8] | 80.8 [78.5-82.8] |
| PubMedQA | 71.4 [67.3-75.2] | 57.0 [52.6-61.3] | 60.8 [56.5-65.0] |
| MedMCQA | 60.1 [57.0-63.1] | 68.2 [65.2-71.0] | 66.4 [63.4-69.3] |
| GPQA Diamond | 43.4 [36.7-50.4] | 49.5 [42.6-56.4] | 48.5 [41.6-55.4] |
| MMLU Med | 80.1 [77.2-82.7] | 90.6 [88.4-92.4] | 87.9 [85.5-90.0] |
| GSM8K (8-shot) | 84.7 [78.0-89.6] | 97.3 [93.3-99.0] | 69.3 [61.5-76.2] |

(Gemma-4-26B-A4B is scored in its chat template; Qwen models and ours use raw-text scoring. Its low GSM8K score comes from the raw 8-shot format and is probably a
protocol effect, not a measure of its math ability.)

### Gap to our model (generative, points; `*` = 95% intervals do not overlap)

| Benchmark | Qwen3.8-27B-FP8 | Gemma-4-26B-A4B |
|---|---|---|
| MedQA (USMLE) | +12.5 * | +17.8 * |
| PubMedQA | -4.4 | -2.2 |
| MedMCQA | +10.0 * | +15.6 * |
| GPQA Diamond | +3.5 | +28.3 * |
| MMLU Med | +8.6 * | +9.6 * |

### What this shows

- **Larger models are clearly better on knowledge benchmarks.** On MedQA, MedMCQA and MMLU Med the two larger models lead the 4B model by roughly 9 to 18 points,
  with non-overlapping intervals. A 4B model trained with this pipeline does not close that gap.
- **PubMedQA is the exception.** This model (75.4%) is statistically tied with both larger models (71.0% and 73.2%), subject to the training-overlap caveat above.
- **Reasoning length differs by one to two orders of magnitude.** Our model answers in about 60 to 110 tokens; Gemma-4-26B-A4B uses 300 to 940 and Qwen3.8-27B uses 1,000 to 4,200.
  Gemma-4-26B-A4B only activates about 4B parameters per token, so its per-token compute is comparable to a 4B model, but it needs the memory of a 26B model.
- **Qwen3.8-27B's accuracy is understated.** Its format-valid rate is 87 to 88% on MedQA and MedMCQA and only 48% on GPQA, so some correct answers were lost to the 6,000-token limit.
  The GPQA gap between it and our model (+3.5, not significant) is not meaningful for that reason.
- **GPQA Diamond:** Gemma-4-26B-A4B is far ahead (70.2% vs 41.9%) with 93% valid answers; that gap is real under this harness.

## Benchmark overlap (contamination)

**Method.** A prompt or training row counts as overlapping if it contains a 30-word window (lower-cased, alphanumerics only) that also occurs in a
benchmark test question, or if one of its lines equals a test question exactly. The benchmark index covers the MedQA test set (1,273), MedMCQA validation (4,183),
PubMedQA-labeled (1,000, question and abstract), four MMLU medical subjects (810), GPQA Diamond (198) and GSM8K test (1,319). This finds verbatim
and near-verbatim copies. It does not find paraphrased or reformatted copies.

**Measured**

| Data | Overlapping | Detail |
|---|---|---|
| RLVR prompt pool (used for the released RLVR stage), rebuilt from the same sources for the audit | **50 of 192,248 unique prompts (0.03%)** | multiple-choice 47 of 51,285 (0.09%), PubMedQA-style 0 of 19,999, math 0 of 27,441, general 3 of 93,523. By source: MedQA-train 46 of 10,178 (0.45%), medical-o1 3 of 19,704, MedMCQA-train 1 of 39,989 |

**Not measured**
- The SFT training data (the data behind the released model's SFT stage) was **not** checked with this filter.
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
- **Run-to-run variation.** The RLVR model was evaluated twice with the same settings. Most numbers moved by 0.1 to 0.2 points, but GPQA Diamond
  moved from 45.5% to 41.9% (7 questions out of 198), probably from non-deterministic batching in vLLM. Differences of a few points on small benchmarks
  are within this noise.
- **Contamination was audited only for verbatim matches, and not for all data.** See the Benchmark overlap section: the RLVR prompt pool is 99.97% clean at 30 words, but the SFT A data and the DAPT corpus were not audited.
- **Short reasoning.** The efficiency reward drove reasoning down to roughly the minimum allowed length. This is fine on exam-style
  questions but may be a weakness on problems that need long, multi-step reasoning (see GPQA).
- **No safety tuning for clinical use.** The model can state incorrect facts confidently and was not evaluated for bias, calibration,
  or harmful-advice behaviour.
- **Single training run, single seed.** Differences of a few points between stages are within noise.
- **English only**, and text only.
