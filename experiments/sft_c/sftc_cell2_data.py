from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

CACHE_TRAIN_EVAL = os.path.join(CACHE_DIR, "split")
if os.path.isdir(CACHE_TRAIN_EVAL) and not REBUILD_CACHE:
    print("Cache exists, skipping the build:", CACHE_TRAIN_EVAL)
    print(open(os.path.join(CACHE_DIR, "stats.json")).read() if os.path.exists(os.path.join(CACHE_DIR, "stats.json")) else "")
else:
    t_start = time.perf_counter()
    if MAX_COPIES != 1:
        print(f"[warn] this fast builder makes a single pass over each source, so MAX_COPIES={MAX_COPIES} has no effect (behaves as 1).")
    data_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    # ============================================================ contamination index
    NG = 30                      # long window: 13-grams matched boilerplate (vitals lines, "which of the following..." stems)
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

        def gsm8k():
            ds = load_dataset("openai/gsm8k", "main", split="test")
            for it in ds: index_eval_text(it["question"])
            return len(ds)

        for name, fn in [("MedQA-test", medqa), ("MedMCQA-val", medmcqa), ("PubMedQA-labeled", pubmedqa),
                         ("MMLU-med", mmlu), ("GPQA-Diamond", gpqa), ("GSM8K-test", gsm8k)]:
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

    # ============================================================ sources (SFT-B mix)
    print("Connecting to SFT streams...")
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

    def clean_string(val):
        return val.strip() if isinstance(val, str) else ""

    def format_med_o1(ex):
        q, cot, res = clean_string(ex.get("Question")), clean_string(ex.get("Complex_CoT")), clean_string(ex.get("Response"))
        content = f"<think>\n{cot}\n</think>\n\n{res}" if cot else res
        return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": content}]}

    def format_ii_med(ex):
        conv = ex.get("messages") or ex.get("conversations")
        if conv and isinstance(conv, list):
            messages = []
            for turn in conv:
                role = turn.get("role") or turn.get("from")
                role = "assistant" if role in ("gpt", "assistant") else "user"
                messages.append({"role": role, "content": clean_string(turn.get("content") or turn.get("value"))})
            return {"messages": messages}
        prompt = clean_string(ex.get("instruction") or ex.get("question") or ex.get("input"))
        response = clean_string(ex.get("output") or ex.get("response"))
        return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]}

    def format_medqa(ex):
        q = clean_string(ex.get("question"))
        opts = ex.get("options", {})
        ans = clean_string(ex.get("answer_idx")).upper()
        if not q or ans not in ("A", "B", "C", "D"):
            return {"messages": []}
        prompt = (f"Question: {q}\nA. {clean_string(opts.get('A'))}\nB. {clean_string(opts.get('B'))}\n"
                  f"C. {clean_string(opts.get('C'))}\nD. {clean_string(opts.get('D'))}\n"
                  "Provide your diagnostic reasoning and the final answer.")
        return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"The correct answer is {ans}."}]}

    def format_alpacare(ex):
        inst, inp, out = clean_string(ex.get("instruction")), clean_string(ex.get("input")), clean_string(ex.get("output"))
        prompt = f"{inst}\n\nCase Details: {inp}" if inp else inst
        return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": out}]}

    def format_medmcqa(ex):
        ans_letter = {0: "A", 1: "B", 2: "C", 3: "D", "0": "A", "1": "B", "2": "C", "3": "D"}.get(ex.get("cop"))
        if not ans_letter:
            return {"messages": []}
        q = clean_string(ex.get("question"))
        exp = clean_string(ex.get("exp"))
        if len(exp) < 30:                       # empty / one-liner explanations are not reasoning
            exp = ""
        prompt = (f"Question: {q}\nA. {clean_string(ex.get('opa'))}\nB. {clean_string(ex.get('opb'))}\n"
                  f"C. {clean_string(ex.get('opc'))}\nD. {clean_string(ex.get('opd'))}\n"
                  "Provide your clinical reasoning and state the correct option letter.")
        content = f"<think>\n{exp}\n</think>\n\nThe correct answer is {ans_letter}." if exp else f"The correct answer is {ans_letter}."
        return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": content}]}

    def format_pubmedqa(ex):
        ctx_data = ex.get("context", {})
        contexts = ctx_data.get("contexts", []) if isinstance(ctx_data, dict) else []
        ctx_str = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
        q = clean_string(ex.get("question"))
        decision = clean_string(ex.get("final_decision")).lower()
        long_answer = clean_string(ex.get("long_answer"))
        if decision not in ("yes", "no", "maybe"):
            return {"messages": []}
        prompt = (f"Context: {ctx_str}\n\nQuestion: {q}\n\n"
                  "Based on the provided biomedical abstract, evaluate the findings and answer with 'yes', 'no', or 'maybe'.")
        content = (f"<think>\n{long_answer}\n</think>\n\nBased on the study, the answer is {decision}."
                   if long_answer else f"Based on the study, the answer is {decision}.")
        return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": content}],
                "src": f"pubmedqa_{decision}"}

    def format_pubmed_causal(ex):
        inst, inp, out = clean_string(ex.get("instruction")), clean_string(ex.get("input")), clean_string(ex.get("output"))
        prompt = f"Biomedical Study Context:\n{inp}\n\nTask: {inst}" if inp else f"Biomedical Inquiry:\n{inst}"
        return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": out}]}

    def format_gsm8k(ex):
        q, ans = clean_string(ex.get("question")), clean_string(ex.get("answer"))
        if "####" in ans:
            reasoning, final_num = ans.rsplit("####", 1)
            reasoning = re.sub(r"<<.*?>>", "", reasoning).strip()      # drop calculator annotations
            content = f"<think>\n{reasoning}\n</think>\n\nThe answer is {final_num.strip()}."
        else:
            content = re.sub(r"<<.*?>>", "", ans)
        return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": content}]}

    def format_orca(ex):
        return {"messages": [{"role": "user", "content": clean_string(ex.get("question"))},
                             {"role": "assistant", "content": clean_string(ex.get("answer"))}]}

    def format_arc(ex):
        q = clean_string(ex.get("question"))
        ch = ex.get("choices", {})
        labels = ch.get("label", []) if isinstance(ch, dict) else []
        texts = ch.get("text", []) if isinstance(ch, dict) else []
        key = clean_string(ex.get("answerKey"))
        if not key:
            return {"messages": []}
        opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        return {"messages": [{"role": "user", "content": f"Question: {q}\n{opts}\nAnswer directly with the correct choice."},
                             {"role": "assistant", "content": f"The correct answer is {key}."}]}

    def format_ultrachat(ex):
        msgs = [{"role": m.get("role", "user"), "content": clean_string(m.get("content"))}
                for m in ex.get("messages", []) if m.get("content")]
        return {"messages": msgs}

    def prepare_sft_stream(ds, formatter, src):
        def fmt(ex):
            out = formatter(ex)
            out.setdefault("src", src)
            return out
        return (ds.map(fmt).select_columns(["messages", "src"])
                  .filter(lambda x: len(x.get("messages", [])) >= 2 and len(x["messages"][-1]["content"]) > 3))

    sources = [  # (name, stream, formatter, intended share: used only for the report below)
        ("med_o1", med_o1_stream, format_med_o1, 0.16),
        ("ii_med", ii_med_stream, format_ii_med, 0.15),
        ("medqa_train", medqa_stream, format_medqa, 0.11),
        ("medmcqa", medmcqa_stream, format_medmcqa, 0.14),
        ("pubmedqa", pubmedqa_stream, format_pubmedqa, 0.09),
        ("pubmed_causal", pubmed_causal_stream, format_pubmed_causal, 0.05),
        ("alpacare", alpacare_stream, format_alpacare, 0.06),
        ("gsm8k", gsm8k_stream, format_gsm8k, 0.08),
        ("orca_math", orca_math_stream, format_orca, 0.06),
        ("arc", arc_stream, format_arc, 0.04),
        ("ultrachat", ultrachat_stream, format_ultrachat, 0.06),
    ]
    assert abs(sum(s[3] for s in sources) - 1.0) < 1e-9

    # ============================================================ stage 1: one pass per source, all sources in parallel threads
    # (with MAX_COPIES=1 the final set is just the union of the sources, so no interleaving / recycling is needed)
    MAX_CHARS = 16_000     # a row this long cannot fit in TARGET_SEQ_LEN tokens (it would need >= 8 chars/token): skip before tokenizing

    def process_source(name, stream, formatter):
        out, local = [], collections.Counter()
        for rec in prepare_sft_stream(stream, formatter, name):
            messages, src = rec["messages"], rec["src"]
            if sum(len(m["content"]) for m in messages) > MAX_CHARS:
                local["skipped_length"] += 1
                continue
            try:
                text = data_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except Exception:  # noqa: BLE001
                local["skipped_template_error"] += 1
                continue
            n_tok = len(data_tokenizer.encode(text, add_special_tokens=False))
            if not (40 < n_tok <= TARGET_SEQ_LEN):
                local["skipped_length"] += 1
                continue
            key = hashlib.md5(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()
            out.append((key, text, n_tok, src))
        return out, local

    results = {}
    with ThreadPoolExecutor(max_workers=len(sources)) as ex:
        futs = {ex.submit(process_source, n, s, f): n for n, s, f, _ in sources}
        for fut in as_completed(futs):
            name = futs[fut]
            results[name] = fut.result()
            print(f"  [{(time.perf_counter() - t_start) / 60:5.1f} min] finished {name:14s} {len(results[name][0]):>8,} candidate rows", flush=True)

    # ============================================================ stage 2: merge in fixed source order: dedupe -> contamination -> caps
    seen, kept_rows = set(), []
    st = collections.Counter()
    kept_by_src, tokens = collections.Counter(), 0
    for name, *_ in sources:
        out, local = results[name]
        st.update(local)
        for key, text, n_tok, src in tqdm(out, desc=f"merge {name}", leave=False):
            if key in seen:
                st["skipped_duplicate"] += 1
                continue
            seen.add(key)
            if CONTAM_FILTER and contaminated(text):
                st["skipped_contaminated"] += 1
                st[f"contaminated_{src}"] += 1
                continue
            if PUBMED_YES_NO_RATIO_CAP and src == "pubmedqa_yes" and \
                    kept_by_src["pubmedqa_yes"] > PUBMED_YES_NO_RATIO_CAP * max(kept_by_src["pubmedqa_no"], 50):
                st["skipped_pubmed_yes_cap"] += 1
                continue
            kept_rows.append({"text": text})
            kept_by_src[src] += 1
            tokens += n_tok

    ds_all = Dataset.from_list(kept_rows)
    split = ds_all.train_test_split(test_size=HOLDOUT, seed=SEED, shuffle=True)
    split.save_to_disk(CACHE_TRAIN_EVAL)
    stats = {"unique_rows": len(kept_rows), "tokens": tokens, "kept_by_source": dict(kept_by_src), "skips": dict(st),
             "max_copies": MAX_COPIES, "contam_filter": CONTAM_FILTER, "ngram": NG, "pubmed_cap": PUBMED_YES_NO_RATIO_CAP}
    json.dump(stats, open(os.path.join(CACHE_DIR, "stats.json"), "w"), indent=2)

    print(f"\nSFT C cache saved: {CACHE_TRAIN_EVAL}   (built in {(time.perf_counter() - t_start) / 60:.1f} min)")
    print(f"train {len(split['train']):,} | eval {len(split['test']):,} | {tokens / 1e6:.1f}M tokens")
    print("\nkept per source (compare with the intended shares):")
    for name, _, _, p in sources:
        n = sum(v for k, v in kept_by_src.items() if k == name or k.startswith(name + "_"))
        print(f"  {name:14s} intended {p * 100:4.0f}%   kept {n:>8,} ({100 * n / max(len(kept_rows), 1):4.1f}%)")
    print("\nPubMedQA labels kept:", {k: v for k, v in kept_by_src.items() if k.startswith("pubmedqa_")})
    print("skips:", dict(st))
