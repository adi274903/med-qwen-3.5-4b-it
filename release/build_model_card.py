import re
readme = open('README.md').read()
OURS = "Med-Qwen-3.5-4B-it (ours)"

def section(text, start, end):
    i = text.index(start); j = text.index(end, i + len(start)) if end else len(text)
    return text[i:j].rstrip() + "\n"

# ------------------------------------------------------------------ table helpers
def split_row(ln): return [c.strip() for c in ln.strip().strip("|").split("|")]
def join_row(cells): return "| " + " | ".join(cells) + " |"
def drop_columns(lines, names):
    hdr = split_row(lines[0]); keep = [i for i, h in enumerate(hdr) if h.replace("**", "") not in names]
    out = []
    for ln in lines:
        c = split_row(ln) if not set(ln.replace("|", "").strip()) <= set("-: ") else None
        out.append(join_row([c[i] for i in keep]) if c else "|" + "---|" * len(keep))
    return out
def num(cell):
    m = re.search(r"\d+(?:\.\d+)?", cell.replace("**", ""))
    return float(m.group()) if m and "n/a" not in cell and "not run" not in cell else None
def bold_best(lines):
    out = [lines[0], lines[1]]
    for ln in lines[2:]:
        c = split_row(ln); vals = [num(x) for x in c[1:]]; c = [c[0]] + [x.replace("**", "") for x in c[1:]]
        best = max(v for v in vals if v is not None)
        out.append(join_row([c[0]] + [f"**{x}**" if v == best else x for x, v in zip(c[1:], vals)]))
    return out
def process_tables(text, fn, only_after=None):
    lines = text.split("\n"); out = []; tbl = []; ctx = None
    def flush():
        nonlocal tbl
        if tbl: out.extend(fn(tbl, ctx)); tbl = []
    for ln in lines:
        if ln.startswith("###"): ctx = ln
        if ln.startswith("|"): tbl.append(ln)
        else: flush(); out.append(ln)
    flush(); return "\n".join(out)

# ------------------------------------------------------------------ final-model evaluation section (numbers checked against the README below)
EVAL = f"""## Evaluation

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
"""
# every number in the new section must appear in the README tables
for tok in re.findall(r"\d+\.\d+ \[\d+\.\d+-\d+\.\d+\]", EVAL):
    assert tok in readme, f"number not found in README: {tok}"

# ------------------------------------------------------------------ comparison with other ~4B models
C4 = section(readme, "## Comparison with other ~4B models", "## Comparison with larger")
C4 = C4.replace("RLVR (ours)", OURS)
def fn4(tbl, ctx):
    t = drop_columns(tbl, {"SFT (ours)", "Base", "DAPT", "SFT"}) if tbl[0].startswith("| Benchmark") else tbl
    if tbl[0].startswith("| Model"):                      # token table: drop the SFT row
        t = [l for l in tbl if not l.startswith("| SFT (ours)")]
    elif ctx and ("Generative evaluation" in ctx or "Log-likelihood" in ctx):
        t = bold_best(t)
    return t
C4 = process_tables(C4, fn4)
legend = "\n**Bold** = highest point estimate in the row among the ~4B models; most intervals overlap, so treat near-ties as ties.\n"
C4 = C4.replace("\n### Answer length and format validity", legend + "\n### Answer length and format validity", 1)
C4 = C4.replace("\n### Log-likelihood evaluation", "\n**Bold** = fewest generated tokens in the column.\n\n### Log-likelihood evaluation", 1)
C4 = C4.replace("\n### What the comparison shows", legend + "\n### What the comparison shows", 1)
C4 = re.sub(r" RLVR's 45\.5% is not significantly different from SFT's 48\.5% or Gemma-4-E4B's 41\.4%\.", " Our 45.5% is not significantly different from Gemma-4-E4B's 41.4%.", C4)
C4 = C4.replace("the RLVR model (73.5%)", "this model (73.5%)").replace("RLVR (75.2%)", "this model (75.2%)").replace("between RLVR (60.3%)", "between this model (60.3%)")

# ------------------------------------------------------------------ comparison with larger models
C27 = section(readme, "## Comparison with larger", "## Usage").replace("RLVR (ours)", OURS)
C27 = process_tables(C27, lambda tbl, ctx: drop_columns(tbl, {"SFT (ours)"}) if tbl[0].startswith("| Benchmark") and "SFT (ours)" in tbl[0] else tbl)
C27 = C27.replace("Gap to RLVR", "Gap to our model").replace("RLVR numbers here are from a second evaluation pass of\nthe same adapter", "Our numbers here are from a second evaluation pass of\nthe same model")
C27 = C27.replace("the 4B RLVR model", "the 4B model").replace("RLVR answers in about", "Our model answers in about").replace("RLVR (75.4%)", "this model (75.4%)").replace("RLVR's", "our").replace("between it and RLVR", "between it and our model").replace("the RLVR model", "this model")

# ------------------------------------------------------------------ rest
TRAIN = section(readme, "## Training (RLVR stage)", "## Evaluation").replace("**Adapter:** LoRA r=64", "**Trained parameters:** LoRA r=64")
OVERLAP = section(readme, "## Benchmark overlap", "## Limitations")
OVERLAP = OVERLAP.replace("(used for the released RLVR adapter)", "(used for the released RLVR stage)")
OVERLAP = re.sub(r"- The SFT A training data .*?that is an assumption\.", "- The SFT training data (the data behind the released model's SFT stage) was **not** checked with this filter.", OVERLAP, flags=re.S)
LIM = section(readme, "## Limitations", "## Citation")

def drop_blocks(text, words):
    out, block = [], []
    def flush():
        if block and not any(w in " ".join(block) for w in words): out.extend(block)
        block.clear()
    for line in text.splitlines():
        if line.startswith("- "): flush(); block.append(line)
        elif block and line.startswith("  "): block.append(line)
        else: flush(); out.append(line)
    flush(); return "\n".join(out) + "\n"
def drop_refs(text, words):
    paras = []
    for para in text.split("\n\n"):
        if para.lstrip().startswith("|"):
            rows = [r for r in para.splitlines() if not any(w in r for w in words)]
            if len(rows) >= 3: paras.append("\n".join(rows))
        elif para.lstrip().startswith(("#", "```", "- ", ">")) or not any(w in para for w in words): paras.append(para)
    return "\n\n".join(paras)
WORDS = ["ablation", "noDAPT", "no-DAPT", "SFT-C", "SFT C", "RLVR-C", "SFT/DAPT/Base entries", "SFT model"]
BODY = "\n".join([TRAIN, EVAL, C4, C27, OVERLAP, LIM])
BODY = drop_refs(drop_blocks(BODY, WORDS), WORDS)

HIGHLIGHTS = f"""## Highlights

- **Short reasoning, competitive accuracy.** MedQA (USMLE) 73.5% with 66 generated tokens on average; the other recent small models in the comparison need 750 to 4,100 tokens for similar MedQA accuracy.
- **Statistically tied with the best small models on MedQA:** Gemma-4-E4B 75.6% (748 tokens) and Qwen3-4B-Thinking 70.8% (3,792 tokens).
- **PubMedQA 75.2%,** tied with Qwen3-4B-Thinking (75.0%, about 2,500 tokens) and above the other small models.
- **Reliable format:** 99 to 100% of answers are parseable on every benchmark; the long-reasoning models lose 9 to 69% of their answers to the token limit on some sets.
- **Honest limits:** other models score higher on MedMCQA and MMLU Med, and the model is 9 to 18 points behind 26-27B models on knowledge benchmarks.
- **Benchmark overlap:** 0.03% of the RLVR prompt pool matched a benchmark question verbatim; the SFT and DAPT data were not fully audited (details below).

> **Not a medical device.** This is a research model. It can be wrong and must not be used for diagnosis, treatment decisions, or any clinical purpose without expert review.
"""
FRONT = """---
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
"""
USAGE = '''## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "adi274903/Med-Qwen-3.5-4B-it"
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, device_map="auto")

question = (
    "Question: A 54-year-old man presents with ...\\n"
    "A. ...\\nB. ...\\nC. ...\\nD. ...\\n\\n"
    'Think step by step, then finish with "The correct answer is X." where X is A, B, C, or D.'
)
prompt = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
if not prompt.rstrip().endswith("<think>"):
    prompt += "<think>\\n"            # the model was trained to start inside a think block
inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

The output has the form `reasoning ... </think> final answer`. The final answer is either a bare option letter (for example `B`) or a sentence such as
"The correct answer is B."; when scoring, read the text after `</think>` and take the last option letter, `yes`/`no`/`maybe`, or number. vLLM loads this model directly.
'''
card = (FRONT + "\n# Med-Qwen-3.5-4B-it\n\nA 4B medical question-answering model built on Qwen3.5-4B-Base with domain-adaptive pretraining, supervised fine-tuning and "
        "reinforcement learning with verifiable rewards (GRPO), released as one merged checkpoint. It writes a short reasoning trace inside "
        "`<think> ... </think>` and then a final answer.\n\n" + HIGHLIGHTS + "\n" + USAGE + "\n" + BODY)
# ---- final wording fixes
card = card.replace("| Benchmark | RLVR | MedGemma-1.5-4B |", "| Benchmark | **Med-Qwen-3.5-4B-it (ours)** | MedGemma-1.5-4B |")
card = card.replace("completed most GPQA answers, and Our 45.5%", "completed most GPQA answers, and our 45.5%")
card = card.replace("**PubMedQA is the exception.** this model", "**PubMedQA is the exception.** This model")
card = card.replace("where RLVR stands out", "where this model stands out")
gen_legend = "**Bold** = highest point estimate in the row among the ~4B models; most intervals overlap, so treat near-ties as ties."
card = card.replace(gen_legend + "\n\n### Answer length and format validity",
                    gen_legend + " On GPQA Diamond several of the other models are cut off by the token limit (see the validity table below), so that row favours models that answer briefly.\n\n### Answer length and format validity", 1)
assert "RLVR |" not in card.split("### Log-likelihood evaluation (answer scored without reasoning)")[1][:400]
open('README_model_card.md', 'w').write(card)

for bad in ("ablation", "noDAPT", "SFT-C", "Adapter", "<this-repo-id>", "github", "GitHub", "SFT (ours)", "+ SFT", "| SFT |", "DAPT |", "| Base"):
    assert bad not in card, f"still contains {bad!r}"
print(len(card.splitlines()), "lines;", [h for h in card.splitlines() if h.startswith("## ")])
print("remaining 'SFT' lines:"); [print("  ", l.strip()[:150]) for l in card.splitlines() if "SFT" in l]
print("remaining 'adapter' mentions:", card.lower().count("adapter"))
