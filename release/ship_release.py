# =====================================================================================================
# SHIP: merge the full model, smoke-test it, and upload ONLY the merged model to Hugging Face, with a model card
# that carries the benchmarks. One Colab cell (GPU runtime). Needs Drive, an HF token with WRITE access.
#   1) Copy README.md to  /content/drive/MyDrive/README.md   (or change README_SRC); the model card is built from it.
#   2) The repo is created PRIVATE and only made public at the end, if every gate passes.
# =====================================================================================================
import gc, json, os, re, shutil
from pathlib import Path
from huggingface_hub import HfApi, notebook_login, whoami
from safetensors import safe_open

# ---------------------------------------------------------------- settings
MERGED_REPO  = "adi274903/Med-Qwen-3.5-4B-it"                         # the only repo that is uploaded
RLVR_ADAPTER = "/content/drive/MyDrive/Med_Qwen_4B_RLVR_v3_trl/final_adapter"   # DAPT-pipeline RLVR run (300 steps); merged, not uploaded
README_SRC   = "/content/drive/MyDrive/README.md"
BASE_ID      = "Qwen/Qwen3.5-4B-Base"
PRIOR_ADAPTERS = ["adi274903/Med-Qwen-3.5-4B-Base-DAPT-Adapter",        # downloaded and merged first, in this order
                  "adi274903/Med-Qwen-3.5-4B-SFT-Adapter"]
LICENSE = "apache-2.0"           # value written to the model card
LICENSE_CHECKED = False          # set True ONLY after you checked the base model and every training dataset allow this license
MAKE_PUBLIC  = True              # wanted: public. It still waits for LICENSE_CHECKED and the smoke test.
MERGED_DIR   = Path("/content/merged_release")

if not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")

# ---------------------------------------------------------------- 1) pre-flight (nothing is uploaded yet)
adir = Path(RLVR_ADAPTER)
assert adir.is_dir(), f"adapter folder not found: {adir}"
for f in ("adapter_config.json", "adapter_model.safetensors"):
    assert (adir / f).is_file(), f"missing {f} in {adir}"
cfg = json.loads((adir / "adapter_config.json").read_text())
with safe_open(str(adir / "adapter_model.safetensors"), "pt") as sf:
    keys = list(sf.keys())
print(f"RLVR adapter: r={cfg.get('r')} alpha={cfg.get('lora_alpha')} | {len(keys)} tensors | first key: {sorted(keys)[0]}")
assert keys and not any(".language_model." in k for k in keys), "adapter keys contain 'language_model.' (remap first)"
assert cfg.get("r") == 64, f"unexpected LoRA rank {cfg.get('r')}"
assert Path(README_SRC).is_file(), f"README not found: {README_SRC}"
readme = Path(README_SRC).read_text()

# ---------------------------------------------------------------- 2) model card (benchmarks come straight from the README)
def section(text, start, end):
    i = text.index(start)
    j = text.index(end, i + len(start)) if end else len(text)
    return text[i:j].rstrip() + "\n"

def drop_blocks(text, words):
    """Remove bullet blocks ('- ' line plus its indented continuation lines) that mention any of `words`."""
    out, block = [], []
    def flush():
        if block and not any(w in " ".join(block) for w in words):
            out.extend(block)
        block.clear()
    for line in text.splitlines():
        if line.startswith("- "):
            flush(); block.append(line)
        elif block and line.startswith("  "):
            block.append(line)
        else:
            flush(); out.append(line)
    flush()
    return "\n".join(out) + "\n"

BODY = "\n".join([
    section(readme, "## Training (RLVR stage)", "## Evaluation"),
    section(readme, "## Evaluation", "## DAPT ablation"),                 # the ablation section itself is not shown on the card
    section(readme, "## Comparison with other ~4B models", "## Comparison with larger"),
    section(readme, "## Comparison with larger", "## Usage"),
    section(readme, "## Benchmark overlap", "## Limitations"),
    section(readme, "## Limitations", "## Citation"),
])
# wording that names adapters / unreleased variants is rewritten before the filter drops anything
BODY = BODY.replace("(used for the released RLVR adapter)", "(used for the released RLVR stage)")
BODY = BODY.replace("**Adapter:** LoRA r=64", "**Trained parameters:** LoRA r=64")
BODY = re.sub(r"- The SFT A training data .*?that is an assumption\.",
              "- The SFT training data (the data behind the released model's SFT stage) was **not** checked with this filter.", BODY, flags=re.S)
BODY = re.sub(r" Whether DAPT is worth keeping is tested in the \[DAPT ablation\].*?confidence intervals\.", "", BODY, flags=re.S)
# unreleased variants and the ablation are not part of this card
def drop_refs(text, words):
    """Remove table rows and plain paragraphs that mention any of `words` (bullets are handled by drop_blocks)."""
    paras = []
    for para in text.split("\n\n"):
        if para.lstrip().startswith("|"):                      # table: drop only the matching rows
            rows = [r for r in para.splitlines() if not any(w in r for w in words)]
            if len(rows) >= 3:
                paras.append("\n".join(rows))
        elif para.lstrip().startswith(("#", "```", "- ", ">")) or not any(w in para for w in words):
            paras.append(para)
    return "\n\n".join(paras)

WORDS = ["ablation", "noDAPT", "no-DAPT", "SFT-C", "SFT C", "RLVR-C"]
BODY = drop_refs(drop_blocks(BODY, WORDS), WORDS)
BODY = BODY.replace("the same adapter", "the same model")
HIGHLIGHTS = """## Highlights

- **About 7x fewer reasoning tokens at the same accuracy.** MedQA (USMLE): 73.5% with 66 generated tokens on average, against 73.2% with 456 tokens for the SFT model it started from.
- **PubMedQA +13.4 points** over the SFT model (61.8% to 75.2%, intervals do not overlap), and tied with Qwen3-4B-Thinking (75.0%), which needs about 2,500 tokens per answer.
- **Competitive with the best small models, with far shorter output.** MedQA 73.5% is statistically tied with Gemma-4-E4B (75.6%, 748 tokens) and Qwen3-4B-Thinking (70.8%, 3,792 tokens).
- **Reliable format:** 99 to 100% of answers are parseable on every benchmark; the long-reasoning models lose 9 to 69% of their answers to the token limit on some sets.
- **Honest limits:** no gain on MedMCQA or GPQA Diamond over the SFT model (differences are inside the confidence intervals), and 9 to 18 points behind 26-27B models on knowledge benchmarks.
- **Benchmark overlap:** 0.03% of the RLVR prompt pool matched a benchmark question verbatim; SFT and DAPT data were not fully audited (details below).

> **Not a medical device.** This is a research model. It can be wrong and must not be used for diagnosis, treatment decisions, or any clinical purpose without expert review.
"""
FRONT = f"""---
language:
- en
license: {LICENSE}
library_name: transformers
base_model: {BASE_ID}
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
USAGE = f"""## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "{MERGED_REPO}"
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, device_map="auto")

question = (
    "Question: A 54-year-old man presents with ...\\n"
    "A. ...\\nB. ...\\nC. ...\\nD. ...\\n\\n"
    'Think step by step, then finish with "The correct answer is X." where X is A, B, C, or D.'
)
prompt = tok.apply_chat_template([{{"role": "user", "content": question}}], tokenize=False, add_generation_prompt=True)
if not prompt.rstrip().endswith("<think>"):
    prompt += "<think>\\n"            # the model was trained to start inside a think block
inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

The output has the form `reasoning ... </think> final answer`. The final answer is either a bare option letter (for example `B`) or a sentence such as
"The correct answer is B."; when scoring, read the text after `</think>` and take the last option letter, `yes`/`no`/`maybe`, or number. vLLM loads this model directly.
"""
card = (FRONT + "\n# Med-Qwen-3.5-4B-it\n\nA 4B medical question-answering model built on Qwen3.5-4B-Base with domain-adaptive pretraining, supervised fine-tuning and "
        "reinforcement learning with verifiable rewards (GRPO), released as one merged checkpoint. It writes a short reasoning trace inside "
        "`<think> ... </think>` and then a final answer.\n\n" + HIGHLIGHTS + "\n" + USAGE + "\n" + BODY)
for bad in ("<this-repo-id>", "{REPO", "ablation", "noDAPT", "SFT-C", "-Adapter", "<your-"):
    assert bad not in card, f"model card still contains {bad!r}"
assert "| MedQA (USMLE)" in card and "## Highlights" in card
print(f"model card: {len(card.splitlines())} lines; mentions of 'adapter': {card.lower().count('adapter')}")

# ---------------------------------------------------------------- 3) merge: Base + DAPT + SFT (from the Hub) + RLVR (Drive)
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(BASE_ID)
model = AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=torch.bfloat16, device_map="auto")
for src in PRIOR_ADAPTERS + [str(adir)]:
    print("merging", src)
    model = PeftModel.from_pretrained(model, src).merge_and_unload()
if MERGED_DIR.exists():
    shutil.rmtree(MERGED_DIR)
model.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="5GB")
tok.save_pretrained(MERGED_DIR)
(MERGED_DIR / "README.md").write_text(card)
del model; gc.collect(); torch.cuda.empty_cache()
print("merged model saved:", sorted(p.name for p in MERGED_DIR.iterdir()))

# ---------------------------------------------------------------- 4) smoke test the SAVED merged model (fresh load from disk)
tok = AutoTokenizer.from_pretrained(MERGED_DIR)
model = AutoModelForCausalLM.from_pretrained(MERGED_DIR, dtype=torch.bfloat16, device_map="auto")
q = ("Question: A 54-year-old man presents with crushing chest pain radiating to the left arm and diaphoresis. "
     "ECG shows ST elevation in leads II, III and aVF. Which artery is most likely occluded?\n"
     "A. Left anterior descending\nB. Right coronary artery\nC. Left circumflex\nD. Left main\n\n"
     'Think step by step, then finish with "The correct answer is X." where X is A, B, C, or D.')
prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
if not prompt.rstrip().endswith("<think>"):
    prompt += "<think>\n"
inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
text = tok.decode(model.generate(**inputs, max_new_tokens=256, do_sample=False)[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("\n--- smoke test output ---\n" + text)
after = text.split("</think>")[-1] if "</think>" in text else ""
SMOKE_OK = bool(re.search(r"\b[A-D]\b", after))            # closed think block, then an option letter (bare "B" or "The correct answer is B.")
print("smoke test:", "OK (closed </think> and an option letter present; expected B)" if SMOKE_OK else "FAILED: will not be made public")
del model; gc.collect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------- 5) upload (private first)
notebook_login()
print("logged in as:", whoami()["name"])
api = HfApi()
api.create_repo(MERGED_REPO, repo_type="model", private=True, exist_ok=True)
api.upload_folder(folder_path=str(MERGED_DIR), repo_id=MERGED_REPO, repo_type="model",
                  commit_message="Med-Qwen-3.5-4B: merged model with model card")
print(f"uploaded (private): https://huggingface.co/{MERGED_REPO}")

# ---------------------------------------------------------------- 6) publish only if every gate passes
if MAKE_PUBLIC and SMOKE_OK and LICENSE_CHECKED:
    api.update_repo_settings(MERGED_REPO, private=False)
    print(f"PUBLIC: https://huggingface.co/{MERGED_REPO}")
else:
    why = [n for n, ok in (("MAKE_PUBLIC is False", MAKE_PUBLIC), ("smoke test failed", SMOKE_OK),
                           ("LICENSE_CHECKED is False (check the base model and every training dataset license, then set it True)", LICENSE_CHECKED)) if not ok]
    print("The repo is still PRIVATE because: " + "; ".join(why))
    print("After fixing, make it public with:\n"
          f"    HfApi().update_repo_settings('{MERGED_REPO}', private=False)")
