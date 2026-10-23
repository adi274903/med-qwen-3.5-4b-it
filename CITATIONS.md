# Citations and acknowledgements

Everything this project builds on. BibTeX for every entry is in [`references.bib`](references.bib); the keys are given in brackets.
Datasets and models keep their own licenses and terms, which are set by their authors on the linked pages. They are not redistributed here.

## Citing this project
```bibtex
@misc{sannabhadti2026medqwen,
  title  = {Med-Qwen-3.5-4B-it: a 4B medical question-answering model with short reasoning (DAPT, SFT, GRPO with verifiable rewards)},
  author = {Sannabhadti, Aditya},
  year   = {2026},
  url    = {https://github.com/adi274903/med-qwen-3.5-4b-it},
  note   = {Model: https://huggingface.co/adi274903/Med-Qwen-3.5-4B-it}
}
```
Please also cite the base model and any dataset or benchmark whose numbers you report.

## Base model and architecture
| What | Reference |
|---|---|
| Base model: [`Qwen/Qwen3.5-4B-Base`](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | Qwen Team, model card (2026) [`qwen35base`]; Qwen3 Technical Report, Yang et al., 2025, [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) [`yang2025qwen3`] |
| Gated DeltaNet linear-attention layers in Qwen3.5 | Gated Delta Networks, Yang, Kautz and Hatamizadeh, 2024, [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) [`yang2024gateddeltanet`] |
| Triton kernels for those layers | [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) [`yang2024fla`] |

## Training methods
| Method | Reference |
|---|---|
| Domain-adaptive pretraining (DAPT) | Don't Stop Pretraining, Gururangan et al., 2020, [arXiv:2004.10964](https://arxiv.org/abs/2004.10964) [`gururangan2020dontstop`] |
| LoRA | Hu et al., 2021, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) [`hu2022lora`] |
| DoRA (used in the earlier Instruct release and the DAPT ablation) | Liu et al., 2024, [arXiv:2402.09353](https://arxiv.org/abs/2402.09353) [`liu2024dora`] |
| GRPO | DeepSeekMath, Shao et al., 2024, [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) [`shao2024deepseekmath`] |
| RLVR (reinforcement learning with verifiable rewards) | Tulu 3, Lambert et al., 2024, [arXiv:2411.15124](https://arxiv.org/abs/2411.15124) [`lambert2024tulu3`]; also DeepSeek-R1, 2025, [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) [`deepseek2025r1`] |

## Training data
Each row lists where the dataset is used in this repository.

| Dataset | Used in | Reference |
|---|---|---|
| [`FreedomIntelligence/medical-o1-reasoning-SFT`](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) | SFT; RLVR prompt pool | HuatuoGPT-o1, Chen et al., 2024, [arXiv:2412.18925](https://arxiv.org/abs/2412.18925) [`chen2024huatuogpto1`] |
| [`Intelligent-Internet/II-Medical-Reasoning-SFT`](https://huggingface.co/datasets/Intelligent-Internet/II-Medical-Reasoning-SFT) | SFT; RLVR prompt pool | Dataset card, Intelligent Internet [`iimedical`] |
| [`medalpaca/medical_meadow_medqa`](https://huggingface.co/datasets/medalpaca/medical_meadow_medqa) (MedQA questions) | SFT | MedQA, Jin et al., 2020, [arXiv:2009.13081](https://arxiv.org/abs/2009.13081) [`jin2021medqa`]; Medical Meadow / MedAlpaca, Han et al., 2023, [arXiv:2304.08247](https://arxiv.org/abs/2304.08247) [`han2023medalpaca`]; dataset card [`medicalmeadowmedqa`] |
| [`lavita/AlpaCare-MedInstruct-52k`](https://huggingface.co/datasets/lavita/AlpaCare-MedInstruct-52k) | SFT; RLVR prompt pool | AlpaCare, Zhang et al., 2023, [arXiv:2310.14558](https://arxiv.org/abs/2310.14558) [`zhang2023alpacare`] |
| [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | SFT; RLVR prompt pool | UltraChat, Ding et al., 2023, [arXiv:2305.14233](https://arxiv.org/abs/2305.14233) [`ding2023ultrachat`] |
| [`microsoft/orca-math-word-problems-200k`](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | SFT; RLVR prompt pool | Orca-Math, Mitra et al., 2024, [arXiv:2402.14830](https://arxiv.org/abs/2402.14830) [`mitra2024orcamath`] |
| [`GBaker/MedQA-USMLE-4-options`](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) (train split) | RLVR prompt pool | MedQA, Jin et al., 2020, [arXiv:2009.13081](https://arxiv.org/abs/2009.13081) [`jin2021medqa`] |
| [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa) (train split) | RLVR prompt pool | MedMCQA, Pal et al., 2022, [arXiv:2203.14371](https://arxiv.org/abs/2203.14371) [`pal2022medmcqa`] |
| [`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA) (`pqa_artificial`) | RLVR prompt pool | PubMedQA, Jin et al., 2019, [arXiv:1909.06146](https://arxiv.org/abs/1909.06146) [`jin2019pubmedqa`] |
| [`medalpaca/medical_meadow_pubmed_causal`](https://huggingface.co/datasets/medalpaca/medical_meadow_pubmed_causal) | RLVR prompt pool | Medical Meadow, Han et al., 2023, [arXiv:2304.08247](https://arxiv.org/abs/2304.08247) [`han2023medalpaca`]; dataset card [`medicalmeadowcausal`] |
| [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) (train split) | RLVR prompt pool | GSM8K, Cobbe et al., 2021, [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) [`cobbe2021gsm8k`] |
| [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc) (ARC-Challenge, train split) | RLVR prompt pool | ARC, Clark et al., 2018, [arXiv:1803.05457](https://arxiv.org/abs/1803.05457) [`clark2018arc`] |
| [`intronhealth/afrimedqa_v2`](https://huggingface.co/datasets/intronhealth/afrimedqa_v2) | RLVR prompt pool | AfriMed-QA, Olatunji et al., 2024, [arXiv:2411.15640](https://arxiv.org/abs/2411.15640) [`olatunji2024afrimedqa`] |

The DAPT corpus is not listed because its build code is not part of this repository; add it here when it is. The `experiments/sft_c` ablation uses the sources above with
a different SFT mix.

## Evaluation benchmarks
| Benchmark | Split used | Reference |
|---|---|---|
| MedQA (USMLE, 4 options), [`GBaker/MedQA-USMLE-4-options`](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) | test | Jin et al., 2020, [arXiv:2009.13081](https://arxiv.org/abs/2009.13081) [`jin2021medqa`] |
| MedMCQA, [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa) | validation | Pal et al., 2022, [arXiv:2203.14371](https://arxiv.org/abs/2203.14371) [`pal2022medmcqa`] |
| PubMedQA, [`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA) | `pqa_labeled` | Jin et al., 2019, [arXiv:1909.06146](https://arxiv.org/abs/1909.06146) [`jin2019pubmedqa`] |
| MMLU, [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) (clinical knowledge, college medicine, medical genetics, professional medicine) | test | Hendrycks et al., 2020, [arXiv:2009.03300](https://arxiv.org/abs/2009.03300) [`hendrycks2021mmlu`] |
| GPQA Diamond, [`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa) | `gpqa_diamond` | Rein et al., 2023, [arXiv:2311.12022](https://arxiv.org/abs/2311.12022) [`rein2023gpqa`] |
| GSM8K (8-shot), [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | test | Cobbe et al., 2021, [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) [`cobbe2021gsm8k`] |

## Models used for comparison
| Model | Reference |
|---|---|
| [`google/medgemma-1.5-4b-it`](https://huggingface.co/google/medgemma-1.5-4b-it) | Google, model card [`medgemma15`]; MedGemma Technical Report, Sellergren et al., 2025, [arXiv:2507.05201](https://arxiv.org/abs/2507.05201) [`sellergren2025medgemma`] |
| [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it), [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) | Google, model cards [`gemma4e4b`], [`gemma426b`] |
| [`Qwen/Qwen3-4B-Thinking-2507`](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507), [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B), [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | Qwen Team, model cards [`qwen3thinking2507`], [`qwen35`], [`qwen38`]; Qwen3 Technical Report [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) [`yang2025qwen3`] |
| [`mistralai/Ministral-3-3B-Reasoning-2512`](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512) | Mistral AI, model card [`ministral3`] |

## Software
[Hugging Face Transformers](https://arxiv.org/abs/1910.03771) [`wolf2020transformers`], [Datasets](https://arxiv.org/abs/2109.02846) [`lhoest2021datasets`], [TRL](https://github.com/huggingface/trl) [`vonwerra2020trl`],
[PEFT](https://github.com/huggingface/peft) [`peft`], [Unsloth](https://github.com/unslothai/unsloth) [`unsloth`] (used for SFT only), and
[vLLM](https://arxiv.org/abs/2309.06180), Kwon et al., 2023 [`kwon2023vllm`] (rollouts and evaluation).

## Statistics
95% Wilson score intervals: Wilson, *J. Am. Stat. Assoc.* 22(158), 1927 [`wilson1927`]. Exact McNemar test for the matched trap-question pairs: McNemar, *Psychometrika* 12(2), 1947 [`mcnemar1947`].
