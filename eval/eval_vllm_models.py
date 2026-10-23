"""
Multi-model evaluation with vLLM (+ optional API models): same protocol for every model.
Paste into ONE Colab cell. Edit MODELS below; finished models are skipped (results live on Drive).

 1) logprob   : summed log-likelihood of each answer continuation (+ 8-shot GSM8K)       [open-weight models]
 2) generative: chat template (+ forced <think> for thinking models), greedy, same answer parser  [all models]
"""
import csv
import gc
import json
import math
import os
import random
import re
import sys
from pathlib import Path

# =========================================================================
# 1. Configuration
# =========================================================================
BASE_MODEL_ID = "Qwen/Qwen3.5-4B-Base"
DAPT = "adi274903/Med-Qwen-3.5-4B-Base-DAPT-Adapter"
SFT = "adi274903/Med-Qwen-3.5-4B-SFT-Adapter"
RLVR_RUN = "/content/drive/MyDrive/Med_Qwen_4B_RLVR_v3_trl"
RLVR_C = "/content/drive/MyDrive/Med_Qwen_4B_RLVR_SFTC_noDAPT/final_adapter"   # RLVR on Base + SFT C
SFT_C = "/content/drive/MyDrive/Med_Qwen_4B_SFT_C_noDAPT/final_adapter_peft_fixed"   # SFT C: base -> SFT, no DAPT (run notebook cell 6 first)

RESULTS_DIR = Path(RLVR_RUN + "/benchmark_results_vllm")     # same folder as before: finished models are skipped
MERGED_ROOT = Path("/content/merged_stage_models")        # local disk; ~8.6 GB per merged model

# Every model to evaluate: name -> {"base": HF id/path, "adapters": [merged in order], "tokenizer": optional}
#  - adapters == []  -> the base is already a complete model, used as is
#  - adapters != []  -> base + adapters are merged on CPU (earlier merged models are reused as a starting point)
MODELS = {
    # ---- ours (results already on Drive; kept so they appear in the comparison tables) ----
    "Base":  {"base": BASE_MODEL_ID, "adapters": []},
    "DAPT":  {"base": BASE_MODEL_ID, "adapters": [DAPT]},
    "SFT":   {"base": BASE_MODEL_ID, "adapters": [DAPT, SFT]},
    "RLVR":  {"base": BASE_MODEL_ID, "adapters": [DAPT, SFT, RLVR_RUN + "/final_adapter"]},
    "SFT-C-noDAPT": {"base": BASE_MODEL_ID, "adapters": [SFT_C]},       # new SFT-B-style data on the plain base
    "RLVR-C-noDAPT": {"base": BASE_MODEL_ID, "adapters": [SFT_C, RLVR_C]},   # merged in order: SFT C, then RLVR

    # ---- other ~4B models (gated ones: log in to Hugging Face and accept the terms) ----
    # think=False : no forced <think>; the answer is parsed from the whole reply
    # think=True  : forced "<think>\n" prefix, answer must come after </think>   (Qwen-style reasoning models)
    # max_new / max_len : generation budget for models that think for a long time
    "MedGemma-1.5-4B": {"base": "google/medgemma-1.5-4b-it", "adapters": [], "think": False, "logprob_chat": True,
                        "llm_kwargs": {"limit_mm_per_prompt": {"image": 0}}},
    "Gemma-4-E4B":     {"base": "google/gemma-4-E4B-it", "adapters": [], "think": False, "logprob_chat": True,
                        "llm_kwargs": {"limit_mm_per_prompt": {"image": 0}}},
    # Mistral-family 4B (3.4B LM + vision encoder), reasoning model: uses Mistral's own tokenizer/format, thinks in [THINK]..[/THINK]
    "Ministral-3-3B-Reasoning": {"base": "mistralai/Ministral-3-3B-Reasoning-2512", "adapters": [], "think": False,
                                 "logprob": False, "chat_api": True, "max_new": 6000, "max_len": 8192,
                                 "llm_kwargs": {"tokenizer_mode": "mistral", "config_format": "mistral", "load_format": "mistral",
                                                "limit_mm_per_prompt": {"image": 0}}},
    # chain-of-thought 4B models
    "Qwen3.5-4B":              {"base": "Qwen/Qwen3.5-4B", "adapters": [], "max_new": 6000, "max_len": 8192},
    "Qwen3-4B-Thinking-2507":  {"base": "Qwen/Qwen3-4B-Thinking-2507", "adapters": [], "max_new": 6000, "max_len": 8192},
    # "Phi-4-mini-reasoning": {"base": "microsoft/Phi-4-mini-reasoning", "adapters": [], "max_new": 6000, "max_len": 8192},  # math-focused
}
SKIP_IF_DONE = True                # skip a model whose stage_<name>.json already exists (delete the file to redo it)
ONLY = ["SFT-C-noDAPT", "RLVR-C-noDAPT"]   # None = everything in MODELS

SEED = 3407
LIMITS = {"MedQA (USMLE)": 1273, "PubMedQA": 500, "MedMCQA": 1000, "GPQA Diamond": 198, "MMLU Med": 810}
MMLU_MED_SUBJECTS = ["clinical_knowledge", "college_medicine", "medical_genetics", "professional_medicine"]
GSM8K_LIMIT = 150
GSM8K_MAX_NEW_TOKENS = 512

RUN_GENERATIVE = True
GEN_STAGES = None                 # None = generative eval for every model except those in NO_GEN
NO_GEN = {"Base", "DAPT"}         # base-style models that never learned the chat/think format
GEN_LIMIT = None                  # None = full sets; or an int to subsample (first N per task)
GEN_MAX_NEW_TOKENS = 1536

MAX_MODEL_LEN = 4096
GPU_MEM_UTIL = 0.85

STAGE_ORDER = list(MODELS)
BENCHMARKS = list(LIMITS)


def chain_key(name):
    m = MODELS[name]
    return [m["base"]] + list(m["adapters"])


def do_logprob(name):
    return MODELS[name].get("logprob", True)


def do_gen(name):
    return RUN_GENERATIVE and name not in NO_GEN and (GEN_STAGES is None or name in GEN_STAGES)


def needs_run(name):
    f = RESULTS_DIR / f"stage_{name}.json"
    if not f.is_file():
        return True
    d = json.loads(f.read_text())
    if do_logprob(name) and any(b not in d.get("logprob", {}) for b in BENCHMARKS):
        return True
    return do_gen(name) and any(b not in d.get("generative", {}) for b in BENCHMARKS)


# =========================================================================
# 2. Pure helpers (no GPU)
# =========================================================================
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
_LETTER_RE = re.compile(r"(?i:answer\s+is|answer\s*:|option|choice(?:\s+is)?)\s*[:\*\(\s]*([A-D])\b")
_LETTER_LINE_RE = re.compile(r"[\*\s\(]*([A-D])\s*(?:[\.\)\*:]|$)")
_YN_RE = re.compile(r"(?i)(?:answer\s+is|answer\s*:)\s*[:\*\s]*(yes|no|maybe)\b")
_YN_LINE_RE = re.compile(r"(?i)\W*(yes|no|maybe)\W*")
NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def split_think(text):
    if text.count(THINK_CLOSE) != 1 or text.count(THINK_OPEN) > 1:
        return None
    if THINK_OPEN in text and text.index(THINK_OPEN) > text.index(THINK_CLOSE):
        return None
    thought, final = text.split(THINK_CLOSE)
    return thought.replace(THINK_OPEN, "").strip(), final.strip()


def _last_line(s):
    lines = [l for l in s.splitlines() if l.strip()]
    return lines[-1].strip() if lines else ""


def parse_choice(final_answer, task):
    if not final_answer:
        return ""
    if task == "mcq":
        m = _LETTER_RE.findall(final_answer)
        if m:
            return m[-1].upper()
        lm = _LETTER_LINE_RE.match(_last_line(final_answer))
        return lm.group(1).upper() if lm else ""
    m = _YN_RE.findall(final_answer)
    if m:
        return m[-1].lower()
    lm = _YN_LINE_RE.fullmatch(_last_line(final_answer))
    return lm.group(1).lower() if lm else ""


def normalize_number(value):
    try:
        number = float(str(value).replace(",", "").strip())
        return format(number, ".12g") if math.isfinite(number) else ""
    except (ValueError, TypeError, OverflowError):
        return ""


def extract_number(text):
    text = text.replace(",", "")
    if "####" in text:
        match = re.search(NUMBER_PATTERN, text.rsplit("####", 1)[-1])
        return normalize_number(match.group()) if match else ""
    explicit = re.findall(rf"(?:the answer is|answer\s*:|equals)\s*[$\\\s]*({NUMBER_PATTERN})",
                          text, flags=re.IGNORECASE)
    if explicit:
        return normalize_number(explicit[-1])
    matches = re.findall(NUMBER_PATTERN, text)
    return normalize_number(matches[-1]) if matches else ""


def wilson_ci(correct, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half) * 100, (centre + half) * 100


def macro_f1(y_true, y_pred, labels):
    f1s = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return 100 * sum(f1s) / len(f1s)


def metrics(y_true, y_pred, labels=None):
    n = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    lo, hi = wilson_ci(correct, n)
    out = {"Accuracy": 100 * correct / n, "CI_low": lo, "CI_high": hi, "N": n}
    if labels:
        out["Macro_F1"] = macro_f1(y_true, y_pred, labels)
    return out


GSM8K_8SHOT_PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.
"""

GEN_INSTR = {
    "mcq": '\n\nThink step by step, then finish with "The correct answer is X." where X is A, B, C, or D.',
    "pubmed": '\n\nBased on the context, think step by step, then finish with "The answer is X." '
              'where X is yes, no, or maybe.',
}


# =========================================================================
# 3. Datasets -> uniform items {"block", "gold", "choices", "task"}
# =========================================================================
def load_items():
    from datasets import load_dataset
    items = {}

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    rows = []
    for it in ds.select(range(min(LIMITS["MedQA (USMLE)"], len(ds)))):
        o = it["options"]
        rows.append({"block": f"Question: {it['question']}\n" + "\n".join(f"{c}. {o[c]}" for c in "ABCD"),
                     "gold": str(it["answer_idx"]).strip().upper()})
    items["MedQA (USMLE)"] = ("mcq", list("ABCD"), rows)

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rows = []
    for it in ds.select(range(min(LIMITS["PubMedQA"], len(ds)))):
        ctx = " ".join(it.get("context", {}).get("contexts", []))
        rows.append({"block": f"Context: {ctx}\nQuestion: {it['question']}",
                     "gold": str(it["final_decision"]).strip().lower()})
    items["PubMedQA"] = ("pubmed", ["yes", "no", "maybe"], rows)

    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    rows = []
    for it in ds.select(range(min(LIMITS["MedMCQA"], len(ds)))):
        cop = int(it["cop"])
        if cop not in range(4):
            continue
        rows.append({"block": f"Question: {it['question']}\nA. {it['opa']}\nB. {it['opb']}\n"
                              f"C. {it['opc']}\nD. {it['opd']}",
                     "gold": "ABCD"[cop]})
    items["MedMCQA"] = ("mcq", list("ABCD"), rows)

    rows = []
    for subj in MMLU_MED_SUBJECTS:
        for it in load_dataset("cais/mmlu", subj, split="test"):
            ch = it["choices"]
            rows.append({"block": f"Question: {it['question']}\n" + "\n".join(f"{c}. {ch[i]}" for i, c in enumerate("ABCD")),
                         "gold": "ABCD"[int(it["answer"])]})
    items["MMLU Med"] = ("mcq", list("ABCD"), rows)

    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        rows = []
        for i, it in enumerate(ds.select(range(min(LIMITS["GPQA Diamond"], len(ds))))):
            opts = [(it["Correct Answer"], True), (it["Incorrect Answer 1"], False),
                    (it["Incorrect Answer 2"], False), (it["Incorrect Answer 3"], False)]
            random.Random(SEED + i).shuffle(opts)
            gold = "ABCD"[next(j for j, (_, ok) in enumerate(opts) if ok)]
            rows.append({"block": f"Question: {it['Question']}\n" +
                                  "\n".join(f"{c}. {t}" for c, (t, _) in zip("ABCD", opts)),
                         "gold": gold})
        items["GPQA Diamond"] = ("mcq", list("ABCD"), rows)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] GPQA Diamond skipped ({type(e).__name__}); log in to Hugging Face and accept the terms.")
    return items


# =========================================================================
# 4. One model: build (merge) -> vLLM -> evaluate -> free the GPU
# =========================================================================
def build_stage_model(stage):
    """Return a path/id of the model for `stage`, merging on CPU and reusing earlier merged models."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = MODELS[stage]
    key = chain_key(stage)
    if not m["adapters"]:
        return m["base"]
    out_dir = MERGED_ROOT / stage
    marker = out_dir / "chain.json"
    if marker.is_file() and json.loads(marker.read_text()) == key:
        print(f"[merge] reusing {out_dir}")
        return str(out_dir)

    start_path, start_key = m["base"], [m["base"]]
    for other in STAGE_ORDER:
        mk = MERGED_ROOT / other / "chain.json"
        if other != stage and mk.is_file():
            c = json.loads(mk.read_text())
            if key[:len(c)] == c and len(c) > len(start_key):
                start_path, start_key = str(MERGED_ROOT / other), c
    todo = key[len(start_key):]
    print(f"[merge] {stage}: start from {start_path}, merging {todo}")

    model = AutoModelForCausalLM.from_pretrained(start_path, torch_dtype=torch.bfloat16,
                                                 device_map="cpu", trust_remote_code=True)
    for adapter in todo:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(m.get("tokenizer", BASE_MODEL_ID), trust_remote_code=True).save_pretrained(out_dir)
    marker.write_text(json.dumps(key))
    del model
    gc.collect()
    return str(out_dir)


def run_stage(stage):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = build_stage_model(stage)
    tok_id = MODELS[stage].get("tokenizer") or (MODELS[stage]["base"] if not MODELS[stage]["adapters"] else BASE_MODEL_ID)
    tokenizer = None if MODELS[stage].get("chat_api") else AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
    items = load_items()
    gsm = None
    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm = gsm.select(range(min(GSM8K_LIMIT, len(gsm))))

    llm = LLM(model=model_path, tokenizer=tok_id,
              dtype="bfloat16", gpu_memory_utilization=GPU_MEM_UTIL, max_model_len=MODELS[stage].get("max_len", MAX_MODEL_LEN),
              enable_prefix_caching=True, seed=SEED, trust_remote_code=True, **MODELS[stage].get("llm_kwargs", {}))

    result = {"logprob": {}, "generative": {}}

    # ---- 1) logprob: summed log-likelihood of " <choice>" after the prompt ------------------------
    lp_params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)

    def logprob_prompt(block, task):
        if task == "pubmed":
            return f"{block}\nBased on the context, answer directly with yes, no, or maybe.\nAnswer:"
        return f"{block}\nAnswer:"

    chat_lp = MODELS[stage].get("logprob_chat", False)   # instruction-tuned non-Qwen models: score inside the chat template

    def chat_prefix(block, task):
        instr = "Answer with only yes, no, or maybe." if task == "pubmed" else "Answer with only the letter of the correct option."
        text = tokenizer.apply_chat_template([{"role": "user", "content": block + "\n\n" + instr}],
                                             tokenize=False, add_generation_prompt=True)
        return tokenizer.encode(text, add_special_tokens=False)          # template already contains BOS

    for name, (task, choices, rows) in (items.items() if do_logprob(stage) else []):
        reqs, spans = [], []
        for r in rows:
            prefix = (chat_prefix(r["block"], task) if chat_lp else
                      tokenizer.encode(logprob_prompt(r["block"], task), add_special_tokens=True))   # adds BOS for Gemma; no-op for Qwen
            for c in choices:
                suffix = tokenizer.encode((c.capitalize() if task == "pubmed" else c) if chat_lp else " " + c, add_special_tokens=False)   # chat models say "Yes", not "yes"
                ids = (prefix + suffix)[-MAX_MODEL_LEN + 2:]
                reqs.append({"prompt_token_ids": ids})
                spans.append(len(suffix))
        outs = llm.generate(reqs, lp_params, use_tqdm=True)
        y_true, y_pred = [], []
        k = 0
        for r in rows:
            scores = {}
            for c in choices:
                o, nsuf = outs[k], spans[k]
                ids = reqs[k]["prompt_token_ids"]
                total = 0.0
                for pos in range(len(ids) - nsuf, len(ids)):
                    total += o.prompt_logprobs[pos][ids[pos]].logprob
                scores[c] = total
                k += 1
            y_pred.append(max(scores, key=scores.get))
            y_true.append(r["gold"])
        result["logprob"][name] = metrics(y_true, y_pred, choices)
        print(f"[{stage}] logprob {name}: {result['logprob'][name]['Accuracy']:.1f}%")

    # ---- GSM8K (8-shot, greedy, stops at the next "Q:") -------------------------------------------
    if do_logprob(stage):
      gp = SamplingParams(max_tokens=GSM8K_MAX_NEW_TOKENS, temperature=0.0, stop=["Q:"])
      prompts = [f"{GSM8K_8SHOT_PROMPT}\nQ: {it['question']}\nA:" for it in gsm]
      outs = llm.generate(prompts, gp, use_tqdm=True)
      golds = [extract_number(it["answer"]) for it in gsm]
      preds = [extract_number(o.outputs[0].text.strip()) for o in outs]
      result["logprob"]["GSM8K"] = metrics(golds, preds)
      print(f"[{stage}] GSM8K: {result['logprob']['GSM8K']['Accuracy']:.1f}%")

    # ---- 2) generative: chat template + <think> ----------------------------------------------------
    if do_gen(stage):
        gen_params = SamplingParams(max_tokens=MODELS[stage].get("max_new", GEN_MAX_NEW_TOKENS), temperature=0.0)

        think = MODELS[stage].get("think", True)

        def chat_ids(user_text):
            text = tokenizer.apply_chat_template([{"role": "user", "content": user_text}],
                                                 tokenize=False, add_generation_prompt=True)
            if think and not text.rstrip().endswith(THINK_OPEN):
                text += THINK_OPEN + "\n"
            return {"prompt_token_ids": tokenizer.encode(text, add_special_tokens=False)}   # template already has BOS if needed

        for name, (task, choices, rows) in items.items():
            sub = rows if GEN_LIMIT is None else rows[:GEN_LIMIT]
            if MODELS[stage].get("chat_api"):     # let vLLM apply the model's own (e.g. Mistral) chat format
                outs = llm.chat([[{"role": "user", "content": r["block"] + GEN_INSTR[task]}] for r in sub],
                                gen_params, use_tqdm=True)
            else:
                outs = llm.generate([chat_ids(r["block"] + GEN_INSTR[task]) for r in sub], gen_params, use_tqdm=True)
            y_true, y_pred, fmt_ok, lens = [], [], 0, []
            for r, o in zip(sub, outs):
                text = o.outputs[0].text
                parts = split_think(text)
                if "[/THINK]" in text:               # Mistral-style reasoning tags
                    parts = (text.split("[/THINK]")[0], text.split("[/THINK]")[-1].strip())
                if think:
                    ok, final = parts is not None, (parts[1] if parts else "")
                else:                                    # no think tags expected: use the final part if present, else everything
                    final = parts[1] if parts else text
                    ok = bool(parse_choice(final, task))
                fmt_ok += ok
                y_pred.append(parse_choice(final, task))
                y_true.append(r["gold"])
                lens.append(len(o.outputs[0].token_ids))
            m = metrics(y_true, y_pred)
            m["Format_OK"] = 100 * fmt_ok / len(sub)
            m["Mean_tokens"] = sum(lens) / len(lens)
            result["generative"][name] = m
            print(f"[{stage}] generative {name}: {m['Accuracy']:.1f}%  fmt {m['Format_OK']:.0f}%  "
                  f"len {m['Mean_tokens']:.0f}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"stage_{stage}.json").write_text(json.dumps(result, indent=2))

    # release the GPU so the next stage can load in this same process
    import torch
    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:  # noqa: BLE001
        print(f"[cleanup] {type(e).__name__}: {e}")
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:  # noqa: BLE001
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()


# =========================================================================
# 5. Comparison tables
# =========================================================================
def cell(m):
    return f"{m['Accuracy']:5.1f} [{m['CI_low']:4.1f}-{m['CI_high']:4.1f}]"


def report():
    res = {}
    for st in STAGE_ORDER:
        f = RESULTS_DIR / f"stage_{st}.json"
        if f.is_file():
            res[st] = json.loads(f.read_text())
    if not res:
        return

    def table(title, key, stages, names):
        stages = [s for s in stages if s in res and res[s].get(key)]
        if not stages:
            return
        print("\n" + "=" * (20 + 21 * len(stages)))
        print(title)
        print(f"{'Benchmark':<16} | " + " | ".join(f"{s[:18]:<18}" for s in stages) +
              "")
        print("-" * (20 + 21 * len(stages)))
        for n in names:
            vals = [res[s][key].get(n) for s in stages]
            if not any(vals):
                continue
            print(f"{n:<16} | " + " | ".join(f"{(cell(v) if v else 'n/a'):<18}" for v in vals))

    table("LOGPROB EVAL: accuracy % [95% Wilson CI]", "logprob", STAGE_ORDER, BENCHMARKS + ["GSM8K"])
    table("GENERATIVE EVAL (chat template + <think>): accuracy % [95% CI]", "generative", STAGE_ORDER, BENCHMARKS)
    for st in STAGE_ORDER:
        if st in res and res[st].get("generative"):
            print(f"\nFormat-valid % / mean tokens ({st}):")
            for n, m in res[st]["generative"].items():
                print(f"  {n:<16} fmt {m['Format_OK']:5.1f}%   len {m['Mean_tokens']:6.0f}")

    rows = []
    for st, d in res.items():
        for kind in ("logprob", "generative"):
            for n, m in d.get(kind, {}).items():
                rows.append({"stage": st, "eval": kind, "benchmark": n, **{k: v for k, v in m.items()}})
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(RESULTS_DIR / "stage_comparison.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {RESULTS_DIR / 'stage_comparison.csv'}")


# =========================================================================
# 6. Run (direct, in this process)
# =========================================================================
if not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

for st in (ONLY or STAGE_ORDER):
    if SKIP_IF_DONE and not needs_run(st):
        print(f"\n[skip] {st}: results already complete")
        continue
    print("\n" + "=" * 100 + f"\nMODEL {st}: " + " + ".join(chain_key(st)), flush=True)
    try:
        run_stage(st)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"[FAILED] {st}: {type(e).__name__}: {e}\n   -> continuing; if the next model fails to load, restart the runtime and rerun (finished models are skipped).")
        gc.collect()

report()
