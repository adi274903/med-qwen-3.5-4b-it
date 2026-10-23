import collections, gc, hashlib, json, os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# ---------------------------------------------------------------- config
TARGET_SEQ_LEN = 2048
LOCAL_ALIGNMENT_CACHE_DIR = "/content/alignment_medical_cache_v4"     # v3 is left untouched
REBUILD_CACHE = False
HOLDOUT = 2500
SEED = 3407
CONTAM_FILTER = True                  # drop prompts that overlap benchmark test questions (MedQA, MedMCQA, PubMedQA, MMLU-med, GPQA, GSM8K)
PUBMED_YES_NO_RATIO_CAP = 2.0         # keeps "yes" <= 2x "no" in the PubMedQA-artificial part (~93% yes otherwise); None = off
TOKENIZER_ID = "Qwen/Qwen3.5-4B-Base"    # same chat template as the merged SFT models; loaded here so this cell is self-contained
MAX_PROMPT_TOKENS = None              # e.g. 512 drops prompts longer than that (the report below counts them either way)
os.makedirs(LOCAL_ALIGNMENT_CACHE_DIR, exist_ok=True)
STATS_PATH = os.path.join(LOCAL_ALIGNMENT_CACHE_DIR, "stats.json")

def _cache_ok():       # an empty or failed earlier build must not count as a cache
    try:
        return json.load(open(STATS_PATH))["unique_rows"] > 0 and os.path.exists(os.path.join(LOCAL_ALIGNMENT_CACHE_DIR, "dataset_dict.json"))
    except Exception:  # noqa: BLE001
        return False

if _cache_ok() and not REBUILD_CACHE:
    print("Cache exists, skipping the build:", LOCAL_ALIGNMENT_CACHE_DIR)
    print(open(STATS_PATH).read() if os.path.exists(STATS_PATH) else "")
else:
    t_start = time.perf_counter()
    import shutil
    shutil.rmtree(LOCAL_ALIGNMENT_CACHE_DIR, ignore_errors=True)
    os.makedirs(LOCAL_ALIGNMENT_CACHE_DIR, exist_ok=True)
    text_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    _probe = text_tokenizer.apply_chat_template([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}], tokenize=False)
    assert "<|im_start|>" in _probe, f"unexpected chat template: {_probe[:80]!r}"

    # ============================================================ contamination index (same as the SFT C builder)
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

    if CONTAM_FILTER:
        build_eval_index()

    # ============================================================ streams (same sources and caps as SFT C)
    print("Connecting to streams...")
    med_o1_stream = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train", streaming=True).take(45_000)
    ii_med_stream = load_dataset("Intelligent-Internet/II-Medical-Reasoning-SFT", split="train", streaming=True).take(50_000)
    medqa_stream = load_dataset("GBaker/MedQA-USMLE-4-options", split="train", streaming=True).take(20_000)
    alpacare_stream = load_dataset("lavita/AlpaCare-MedInstruct-52k", split="train", streaming=True)
    medmcqa_stream = load_dataset("openlifescienceai/medmcqa", split="train", streaming=True).take(50_000)
    pubmedqa_stream = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train", streaming=True).take(30_000)
    pubmed_causal_stream = load_dataset("medalpaca/medical_meadow_pubmed_causal", split="train", streaming=True).take(25_000)
    gsm8k_stream = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    orca_math_stream = load_dataset("microsoft/orca-math-word-problems-200k", split="train", streaming=True).take(25_000)
    arc_stream = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train", streaming=True)
    ultrachat_stream = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).take(20_000)

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

    sources = [  # (name, stream, formatter)
        ("med_o1", med_o1_stream, format_med_o1), ("ii_med", ii_med_stream, format_ii_med),
        ("medqa_train", medqa_stream, format_medqa), ("medmcqa", medmcqa_stream, format_medmcqa),
        ("pubmedqa", pubmedqa_stream, format_pubmedqa), ("pubmed_causal", pubmed_causal_stream, format_pubmed_causal),
        ("alpacare", alpacare_stream, format_alpacare), ("gsm8k", gsm8k_stream, format_gsm8k),
        ("orca_math", orca_math_stream, format_orca), ("arc", arc_stream, format_arc),
        ("ultrachat", ultrachat_stream, format_ultrachat),
    ]

    # ============================================================ stage 1: one pass per source, all sources in parallel threads
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

    if sum(len(r[0]) for r in results.values()) == 0:
        errs = collections.Counter()
        for _, loc in results.values():
            errs.update({k: v for k, v in loc.items() if k.startswith("err:") or k.startswith("skipped")})
        raise RuntimeError(f"no rows survived stage 1; nothing was saved. Counters: {dict(errs)}")

    # ============================================================ stage 2: merge in fixed source order: dedupe -> contamination -> pubmed cap
    seen, kept_rows = set(), []
    st, kept_by_src, tokens = collections.Counter(), collections.Counter(), 0
    for name, *_ in sources:
        out, local = results[name]
        st.update(local)
        for key, row, n_tok, src in tqdm(out, desc=f"merge {name}", leave=False):
            if key in seen:
                st["skipped_duplicate"] += 1
                continue
            seen.add(key)
            if CONTAM_FILTER and contaminated(row["text"]):
                st["skipped_contaminated"] += 1
                st[f"contaminated_{src}"] += 1
                continue
            if PUBMED_YES_NO_RATIO_CAP and src == "pubmedqa_yes" and \
                    kept_by_src["pubmedqa_yes"] > PUBMED_YES_NO_RATIO_CAP * max(kept_by_src["pubmedqa_no"], 50):
                st["skipped_pubmed_yes_cap"] += 1
                continue
            kept_rows.append(row)
            kept_by_src[src] += 1
            tokens += n_tok

    raw_alignment_dataset = Dataset.from_list(kept_rows)
    split_dataset = raw_alignment_dataset.train_test_split(test_size=HOLDOUT, seed=SEED, shuffle=True)
    split_dataset.save_to_disk(LOCAL_ALIGNMENT_CACHE_DIR)
    task_counts = collections.Counter(r["task_type"] for r in kept_rows)
    stats = {"unique_rows": len(kept_rows), "tokens": tokens, "kept_by_source": dict(kept_by_src), "task_types": dict(task_counts),
             "skips": dict(st), "contam_filter": CONTAM_FILTER, "ngram": NG, "pubmed_cap": PUBMED_YES_NO_RATIO_CAP}
    json.dump(stats, open(STATS_PATH, "w"), indent=2)

    print(f"\n✓ Alignment cache saved: {LOCAL_ALIGNMENT_CACHE_DIR}   (built in {(time.perf_counter() - t_start) / 60:.1f} min)")
    print(f"Total Train Samples: {len(split_dataset['train']):,}")
    print(f"Total Eval Samples:  {len(split_dataset['test']):,}")
    print("task types (train+eval):", dict(task_counts))
    print("kept per source:", dict(kept_by_src))
    print("skips:", dict(st))
