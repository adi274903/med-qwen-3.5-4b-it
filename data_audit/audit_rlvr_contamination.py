# Audit: how many prompts of the OLD RLVR pool (cache v3) overlap benchmark test questions?
# v3 is gone, so this rebuilds its contents from the same 12 sources with the same caps (no interleaving needed:
# the pool is the union of the sources), applies the v3 length filter, and COUNTS overlaps with the 30-gram + exact-question
# filter used for v4. Nothing is saved. Needs a runtime with `datasets`, `transformers`; ~5-10 min.
import collections, hashlib, json, os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from transformers import AutoTokenizer

TARGET_SEQ_LEN = 2048
MAX_PROMPT_TOKENS = None
t_start = time.perf_counter()
text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base")

NG = 30                      # long window: 13-grams matched boilerplate ("which of the following...", vitals lines)
words = lambda s: re.findall(r"[a-z0-9]+", s.lower())
EVAL_NGRAMS, EVAL_LINES = set(), set()

def index_eval_text(s):
    w = words(s)
    if len(w) >= NG:
        for i in range(len(w) - NG + 1):
            EVAL_NGRAMS.add(hash(tuple(w[i:i + NG])))
    if len(w) >= 6:
        EVAL_LINES.add(" ".join(w))

def build_eval_index():
    loaded = []

    def safe(name, fn):
        try:
            loaded.append(f"{name} ({fn()})")
        except Exception as e:  # noqa: BLE001
            print(f"[contam] {name} skipped: {type(e).__name__}: {str(e)[:80]}")

    def medqa():
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        for it in ds: index_eval_text(it["question"])
        return len(ds)

    def medmcqa():
        ds = load_dataset("openlifescienceai/medmcqa", split="validation")
        for it in ds: index_eval_text(it["question"])
        return len(ds)

    def pubmedqa():
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        for it in ds:
            index_eval_text(it["question"])
            index_eval_text(" ".join(it["context"]["contexts"]))
        return len(ds)

    def mmlu():
        n = 0
        for subj in ["clinical_knowledge", "college_medicine", "medical_genetics", "professional_medicine"]:
            ds = load_dataset("cais/mmlu", subj, split="test")
            for it in ds: index_eval_text(it["question"])
            n += len(ds)
        return n

    def gpqa():
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        for it in ds: index_eval_text(it["Question"])
        return len(ds)

    def gsm8k_test():
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for it in ds: index_eval_text(it["question"])
        return len(ds)

    for name, fn in [("MedQA-test", medqa), ("MedMCQA-val", medmcqa), ("PubMedQA-labeled", pubmedqa),
                     ("MMLU-med", mmlu), ("GPQA-Diamond", gpqa), ("GSM8K-test", gsm8k_test)]:
        safe(name, fn)
    print(f"[contam] indexed: {', '.join(loaded)} | {len(EVAL_NGRAMS):,} {NG}-grams, {len(EVAL_LINES):,} question lines")

def contaminated(text):
    for line in text.split("\n"):
        l = line.replace("<|im_end|>", "").strip()      # the chat template glues this onto the last line of a message
        if l.lower().startswith("question:"):
            l = l[9:]
        w = words(l)
        if len(w) >= 6 and " ".join(w) in EVAL_LINES:
            return True
    w = words(text)
    for i in range(len(w) - NG + 1):
        if hash(tuple(w[i:i + NG])) in EVAL_NGRAMS:
            return True
    return False


# ---- the v3 sources and caps, exactly as in the old data cell
med_o1_stream = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train", streaming=True).take(35_000)
ii_med_stream = load_dataset("Intelligent-Internet/II-Medical-Reasoning-SFT", split="train", streaming=True).take(35_000)
medqa_stream = load_dataset("GBaker/MedQA-USMLE-4-options", split="train", streaming=True)
afrimedqa_stream = load_dataset("intronhealth/afrimedqa_v2", "default", split="train", streaming=True).take(12_000)
alpacare_stream = load_dataset("lavita/AlpaCare-MedInstruct-52k", split="train", streaming=True).take(25_000)
medmcqa_stream = load_dataset("openlifescienceai/medmcqa", split="train", streaming=True).take(40_000)
pubmedqa_stream = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train", streaming=True).take(20_000)
pubmed_causal_stream = load_dataset("medalpaca/medical_meadow_pubmed_causal", split="train", streaming=True).take(15_000)
gsm8k_stream = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
orca_math_stream = load_dataset("microsoft/orca-math-word-problems-200k", split="train", streaming=True).take(20_000)
arc_stream = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train", streaming=True)
ultrachat_stream = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).take(15_000)

EMPTY_RECORD = {"prompt": [], "ground_truth": "", "task_type": "", "full_target": ""}

def clean_string(val):
    return val.strip() if isinstance(val, str) else ""

# Formatters: identical prompt wording and task types as the earlier RLVR runs (the reward code and the comparison with them depend on it)
def format_med_o1(ex):
    q, cot, res = clean_string(ex.get("Question")), clean_string(ex.get("Complex_CoT")), clean_string(ex.get("Response"))
    if not q or not res: return EMPTY_RECORD
    return {"prompt": [{"role": "user", "content": q}], "ground_truth": "", "task_type": "general_cot", "full_target": f"<think>\n{cot}\n</think>\n\n{res}" if cot else res}

def format_ii_med(ex):
    conv = ex.get("messages") or ex.get("conversations")
    if conv and isinstance(conv, list) and len(conv) >= 2:
        u_msg = clean_string(conv[0].get("content") or conv[0].get("value"))
        a_msg = clean_string(conv[1].get("content") or conv[1].get("value"))
        if u_msg and a_msg:
            return {"prompt": [{"role": "user", "content": u_msg}], "ground_truth": "", "task_type": "general_cot", "full_target": a_msg}
    return EMPTY_RECORD

def format_medqa(ex):
    q, opts = clean_string(ex.get("question")), ex.get("options", {})
    ans = clean_string(ex.get("answer_idx") or ex.get("answer")).upper()
    if not q or ans not in ("A", "B", "C", "D"): return EMPTY_RECORD
    prompt = f"Question: {q}\nA. {clean_string(opts.get('A'))}\nB. {clean_string(opts.get('B'))}\nC. {clean_string(opts.get('C'))}\nD. {clean_string(opts.get('D'))}\nProvide your step-by-step diagnostic reasoning inside <think>...</think> tags and conclude with the correct option letter."
    return {"prompt": [{"role": "user", "content": prompt}], "ground_truth": ans, "task_type": "mcq", "full_target": f"The correct answer is {ans}."}

def format_medmcqa(ex):
    cop_map = {0: "A", 1: "B", 2: "C", 3: "D", "0": "A", "1": "B", "2": "C", "3": "D"}
    ans_letter = cop_map.get(ex.get("cop"))
    if not ans_letter: return EMPTY_RECORD
    q = clean_string(ex.get("question"))
    exp = clean_string(ex.get("exp"))
    if len(exp) < 30: exp = ""                      # empty / one-liner explanations are not reasoning
    prompt = f"Question: {q}\nA. {clean_string(ex.get('opa'))}\nB. {clean_string(ex.get('opb'))}\nC. {clean_string(ex.get('opc'))}\nD. {clean_string(ex.get('opd'))}\nProvide your clinical reasoning inside <think>...</think> tags and state the correct option letter."
    return {"prompt": [{"role": "user", "content": prompt}], "ground_truth": ans_letter, "task_type": "mcq", "full_target": f"<think>\n{exp}\n</think>\n\nThe correct answer is {ans_letter}." if exp else f"The correct answer is {ans_letter}."}

def format_pubmedqa(ex):
    ctx_data = ex.get("context", {})
    contexts = ctx_data.get("contexts", []) if isinstance(ctx_data, dict) else []
    ctx_str = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
    q = clean_string(ex.get("question"))
    decision = clean_string(ex.get("final_decision")).lower()
    long_answer = clean_string(ex.get("long_answer"))
    if decision not in ("yes", "no", "maybe"): return EMPTY_RECORD
    prompt = f"Context: {ctx_str}\n\nQuestion: {q}\n\nBased on the biomedical abstract, reason inside <think>...</think> tags and conclude with 'yes', 'no', or 'maybe'."
    return {"prompt": [{"role": "user", "content": prompt}], "ground_truth": decision, "task_type": "pubmed", "src": f"pubmedqa_{decision}",
            "full_target": f"<think>\n{long_answer}\n</think>\n\nBased on the study, the answer is {decision}." if long_answer else f"Based on the study, the answer is {decision}."}

def format_pubmed_causal(ex):
    inst, inp, out = clean_string(ex.get("instruction")), clean_string(ex.get("input")), clean_string(ex.get("output"))
    if not out: return EMPTY_RECORD
    return {"prompt": [{"role": "user", "content": f"Biomedical Study Context:\n{inp}\n\nTask: {inst}" if inp else f"Biomedical Inquiry:\n{inst}"}], "ground_truth": "", "task_type": "general_cot", "full_target": out}

def format_alpacare(ex):
    inst, inp, out = clean_string(ex.get("instruction")), clean_string(ex.get("input")), clean_string(ex.get("output"))
    if not out: return EMPTY_RECORD
    return {"prompt": [{"role": "user", "content": f"{inst}\n\nCase Details: {inp}" if inp else inst}], "ground_truth": "", "task_type": "general_cot", "full_target": out}

def format_gsm8k(ex):
    q, ans = clean_string(ex.get("question")), clean_string(ex.get("answer"))
    if not q or not ans: return EMPTY_RECORD
    if "####" in ans:
        reasoning, final_num = ans.rsplit("####", 1)
        final_num = final_num.strip().replace(",", "")
        reasoning = re.sub(r"<<.*?>>", "", reasoning).strip()        # drop calculator annotations
        assistant_content = f"<think>\n{reasoning}\n</think>\n\nThe answer is {final_num}."
    else:
        assistant_content = re.sub(r"<<.*?>>", "", ans)
        nums = re.findall(r"[-+]?\d*\.?\d+", ans.replace(",", ""))
        final_num = nums[-1] if nums else ""
    return {"prompt": [{"role": "user", "content": q}], "ground_truth": final_num, "task_type": "math", "full_target": assistant_content}

def format_orca(ex):
    q, a = clean_string(ex.get("question")), clean_string(ex.get("answer"))
    if not q or not a: return EMPTY_RECORD
    return {"prompt": [{"role": "user", "content": q}], "ground_truth": "", "task_type": "math", "full_target": a}

def format_arc(ex):
    q = clean_string(ex.get("question"))
    choices = ex.get("choices", {})
    labels = choices.get("label", []) if isinstance(choices, dict) else []
    texts = choices.get("text", []) if isinstance(choices, dict) else []
    answer_key = clean_string(ex.get("answerKey"))
    if not answer_key or not q: return EMPTY_RECORD
    formatted_choices = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    return {"prompt": [{"role": "user", "content": f"Question: {q}\n{formatted_choices}\nReason inside <think>...</think> and state the correct option letter."}], "ground_truth": answer_key, "task_type": "mcq", "full_target": f"The correct answer is {answer_key}."}

def format_ultrachat(ex):
    raw_msgs = ex.get("messages", [])
    if len(raw_msgs) >= 2:
        u_msg, a_msg = clean_string(raw_msgs[0].get("content")), clean_string(raw_msgs[1].get("content"))
        if u_msg and a_msg:
            return {"prompt": [{"role": "user", "content": u_msg}], "ground_truth": "", "task_type": "general_cot", "full_target": a_msg}
    return EMPTY_RECORD


def format_afrimedqa(ex):
    q = clean_string(ex.get("question") or ex.get("prompt") or ex.get("question_clean"))
    correct_ans = clean_string(ex.get("correct_answer"))
    rationale = clean_string(ex.get("answer_rationale"))
    if not q or not (correct_ans or rationale): return EMPTY_RECORD
    return {"prompt": [{"role": "user", "content": q}], "ground_truth": correct_ans, "task_type": "general_cot",
            "full_target": f"<think>\n{rationale}\n</think>\n\n{correct_ans}" if rationale else correct_ans}

sources = [
    ("med_o1", med_o1_stream, format_med_o1), ("ii_med", ii_med_stream, format_ii_med),
    ("medqa_train", medqa_stream, format_medqa), ("afrimedqa", afrimedqa_stream, format_afrimedqa),
    ("medmcqa", medmcqa_stream, format_medmcqa), ("pubmedqa", pubmedqa_stream, format_pubmedqa),
    ("pubmed_causal", pubmed_causal_stream, format_pubmed_causal), ("alpacare", alpacare_stream, format_alpacare),
    ("gsm8k", gsm8k_stream, format_gsm8k), ("orca_math", orca_math_stream, format_orca),
    ("arc", arc_stream, format_arc), ("ultrachat", ultrachat_stream, format_ultrachat),
]

MAX_CHARS = 16_000     # a row this long cannot fit in TARGET_SEQ_LEN tokens: skip before tokenizing

def process_source(name, stream, formatter):
    out, local = [], collections.Counter()
    for ex in stream:
        rec = formatter(ex)
        if not rec["prompt"] or len(rec["full_target"]) <= 3:
            continue
        if sum(len(m["content"]) for m in rec["prompt"]) + len(rec["full_target"]) > MAX_CHARS:
            local["skipped_length"] += 1
            continue
        try:
            prompt_str = text_tokenizer.apply_chat_template(rec["prompt"], tokenize=False, add_generation_prompt=True)
            full_text = text_tokenizer.apply_chat_template(rec["prompt"] + [{"role": "assistant", "content": rec["full_target"]}],
                                                          tokenize=False, add_generation_prompt=False)
        except Exception as e:  # noqa: BLE001
            local["skipped_template_error"] += 1
            local[f"err: {type(e).__name__}: {str(e)[:100]}"] += 1
            continue
        n_tok = len(text_tokenizer.encode(full_text, add_special_tokens=False))
        if not (40 < n_tok <= TARGET_SEQ_LEN):
            local["skipped_length"] += 1
            continue
        n_prompt = len(text_tokenizer.encode(prompt_str, add_special_tokens=False))
        if n_prompt > 512:
            local["prompt_over_512_tokens"] += 1
            if MAX_PROMPT_TOKENS and n_prompt > MAX_PROMPT_TOKENS:
                local["skipped_prompt_too_long"] += 1
                continue
        key = hashlib.md5(json.dumps([rec["prompt"], rec["full_target"]], ensure_ascii=False).encode()).hexdigest()
        out.append((key, {"prompt": prompt_str, "text": full_text, "ground_truth": rec["ground_truth"], "task_type": rec["task_type"]},
                    n_tok, rec.get("src", name)))
    return out, local

results = {}
with ThreadPoolExecutor(max_workers=len(sources)) as ex_pool:
    futs = {ex_pool.submit(process_source, n, s, f): n for n, s, f in sources}
    for fut in as_completed(futs):
        name = futs[fut]
        results[name] = fut.result()
        print(f"  [{(time.perf_counter() - t_start) / 60:5.1f} min] finished {name:14s} {len(results[name][0]):>8,} candidate rows", flush=True)


build_eval_index()
seen = set()
tot_src, hit_src, tot_task, hit_task = (collections.Counter() for _ in range(4))
examples = []
for name, *_ in sources:
    out, local = results[name]
    for key, row, n_tok, src in out:
        if key in seen:
            continue
        seen.add(key)
        tot_src[name] += 1; tot_task[row["task_type"]] += 1
        if contaminated(row["text"]):
            hit_src[name] += 1; hit_task[row["task_type"]] += 1
            if len(examples) < 5:
                examples.append((name, row["prompt"][:300]))
n, h = sum(tot_src.values()), sum(hit_src.values())
print(f"\nOLD RLVR pool (v3 rebuilt): {h:,} of {n:,} unique prompts overlap a benchmark test question ({100*h/max(n,1):.2f}%)")
print("by source:")
for k in tot_src:
    print(f"  {k:14s} {hit_src[k]:>6,} / {tot_src[k]:>7,}  ({100*hit_src[k]/tot_src[k]:.2f}%)")
print("by task type (these are the rows that get rewards):")
for k in tot_task:
    print(f"  {k:12s} {hit_task[k]:>6,} / {tot_task[k]:>7,}  ({100*hit_task[k]/tot_task[k]:.2f}%)")
for name, e in examples:
    print(f"\n--- example ({name}) ---\n" + repr(e))
