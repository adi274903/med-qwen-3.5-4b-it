import os
import gc
import re
from datasets import load_dataset, interleave_datasets, Dataset
from tqdm.auto import tqdm

TARGET_SEQ_LEN = 2048
LOCAL_ALIGNMENT_CACHE_DIR = "/content/alignment_medical_cache_v3"
os.makedirs(LOCAL_ALIGNMENT_CACHE_DIR, exist_ok=True)

print("Connecting to streams with validated key schemas...")

# 1. Streams
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

# Formatters aligned to verified keys
def format_med_o1(ex):
    q, cot, res = clean_string(ex.get("Question")), clean_string(ex.get("Complex_CoT")), clean_string(ex.get("Response"))
    if not q or not res: return EMPTY_RECORD
    return {"prompt": [{"role": "user", "content": q}], "ground_truth": "", "task_type": "general_cot", "full_target": f"<think>\n{cot}\n</think>\n\n{res}" if cot else res}

def format_ii_med(ex):
    conv = ex.get("messages")
    if conv and isinstance(conv, list) and len(conv) >= 2:
        u_msg = clean_string(conv[0].get("content"))
        a_msg = clean_string(conv[1].get("content"))
        if u_msg and a_msg:
            return {"prompt": [{"role": "user", "content": u_msg}], "ground_truth": "", "task_type": "general_cot", "full_target": a_msg}
    q = clean_string(ex.get("question") or ex.get("problem"))
    if not q: return EMPTY_RECORD
    return {"prompt": [{"role": "user", "content": q}], "ground_truth": "", "task_type": "general_cot", "full_target": "Clinical reasoning validated."}

def format_medqa(ex):
    q, opts = clean_string(ex.get("question")), ex.get("options", {})
    ans = clean_string(ex.get("answer_idx") or ex.get("answer")).upper()
    if not q or not ans or ans not in ("A", "B", "C", "D"): return EMPTY_RECORD
    prompt = f"Question: {q}\nA. {clean_string(opts.get('A'))}\nB. {clean_string(opts.get('B'))}\nC. {clean_string(opts.get('C'))}\nD. {clean_string(opts.get('D'))}\nProvide your step-by-step diagnostic reasoning inside <think>...</think> tags and conclude with the correct option letter."
    return {"prompt": [{"role": "user", "content": prompt}], "ground_truth": ans, "task_type": "mcq", "full_target": f"The correct answer is {ans}."}

def format_afrimedqa(ex):
    q = clean_string(ex.get("question") or ex.get("prompt") or ex.get("question_clean"))
    correct_ans = clean_string(ex.get("correct_answer"))
    rationale = clean_string(ex.get("answer_rationale"))
    if not q or not (correct_ans or rationale): return EMPTY_RECORD
    return {"prompt": [{"role": "user", "content": q}], "ground_truth": correct_ans, "task_type": "general_cot", "full_target": f"<think>\n{rationale}\n</think>\n\n{correct_ans}" if rationale else correct_ans}

def format_medmcqa(ex):
    cop_map = {0: "A", 1: "B", 2: "C", 3: "D", "0": "A", "1": "B", "2": "C", "3": "D"}
    ans_letter = cop_map.get(ex.get("cop"))
    if not ans_letter: return EMPTY_RECORD
    q = clean_string(ex.get("question"))
    exp = clean_string(ex.get("exp"))
    prompt = f"Question: {q}\nA. {clean_string(ex.get('opa'))}\nB. {clean_string(ex.get('opb'))}\nC. {clean_string(ex.get('opc'))}\nD. {clean_string(ex.get('opd'))}\nProvide your clinical reasoning inside <think>...</think> tags and state the correct option letter."
    return {"prompt": [{"role": "user", "content": prompt}], "ground_truth": ans_letter, "task_type": "mcq", "full_target": f"<think>\n{exp}\n</think>\n\nThe correct answer is {ans_letter}." if exp else f"The correct answer is {ans_letter}."}

def format_pubmedqa(ex):
    ctx_data = ex.get("context", {})
    contexts = ctx_data.get("contexts", []) if isinstance(ctx_data, dict) else []
    ctx_str = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
    q = clean_string(ex.get("question"))
    decision = clean_string(ex.get("final_decision")).lower()
    long_answer = clean_string(ex.get("long_answer"))
    if not decision or decision not in ("yes", "no", "maybe"): return EMPTY_RECORD
    prompt = f"Context: {ctx_str}\n\nQuestion: {q}\n\nBased on the biomedical abstract, reason inside <think>...</think> tags and conclude with 'yes', 'no', or 'maybe'."
    return {"prompt": [{"role": "user", "content": prompt}], "ground_truth": decision, "task_type": "pubmed", "full_target": f"<think>\n{long_answer}\n</think>\n\nBased on the study, the answer is {decision}." if long_answer else f"Based on the study, the answer is {decision}."}

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
        reasoning, final_num = ans.split("####")
        final_num = final_num.strip()
        assistant_content = f"<think>\n{reasoning.strip()}\n</think>\n\nThe answer is {final_num}."
    else:
        assistant_content = ans
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
    formatted_choices = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])
    return {"prompt": [{"role": "user", "content": f"Question: {q}\n{formatted_choices}\nReason inside <think>...</think> and state the correct option letter."}], "ground_truth": answer_key, "task_type": "mcq", "full_target": f"The correct answer is {answer_key}."}

def format_ultrachat(ex):
    raw_msgs = ex.get("messages", [])
    if len(raw_msgs) >= 2:
        u_msg = clean_string(raw_msgs[0].get("content"))
        a_msg = clean_string(raw_msgs[1].get("content"))
        if u_msg and a_msg:
            return {"prompt": [{"role": "user", "content": u_msg}], "ground_truth": "", "task_type": "general_cot", "full_target": a_msg}
    return EMPTY_RECORD

def prepare_alignment_stream(ds, formatter):
    return (
        ds.map(formatter)
          .select_columns(["prompt", "ground_truth", "task_type", "full_target"])
          .filter(lambda x: len(x.get("prompt", [])) >= 1 and len(x.get("full_target", "")) > 3)
    )

# 2. Interleave
streams = [
    prepare_alignment_stream(medqa_stream, format_medqa),
    prepare_alignment_stream(med_o1_stream, format_med_o1),
    prepare_alignment_stream(ii_med_stream, format_ii_med),
    prepare_alignment_stream(medmcqa_stream, format_medmcqa),
    prepare_alignment_stream(afrimedqa_stream, format_afrimedqa),
    prepare_alignment_stream(pubmedqa_stream, format_pubmedqa),
    prepare_alignment_stream(gsm8k_stream, format_gsm8k),
    prepare_alignment_stream(orca_math_stream, format_orca),
    prepare_alignment_stream(pubmed_causal_stream, format_pubmed_causal),
    prepare_alignment_stream(alpacare_stream, format_alpacare),
    prepare_alignment_stream(arc_stream, format_arc),
    prepare_alignment_stream(ultrachat_stream, format_ultrachat),
]

probs = [0.15, 0.13, 0.12, 0.12, 0.08, 0.07, 0.08, 0.06, 0.05, 0.05, 0.04, 0.05]

interleaved_alignment_stream = interleave_datasets(
    streams,
    probabilities=probs,
    seed=3407,
    stopping_strategy="all_exhausted"
)

# 3. Generator & Serialization
def alignment_generator():
    pbar = tqdm(desc="Processing Alignment Mixture", unit="samples")
    total_tokens = 0

    for record in interleaved_alignment_stream:
        prompt_msgs = record["prompt"]
        full_target = record["full_target"]

        try:
            prompt_str = text_tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True
            )
            full_msgs = prompt_msgs + [{"role": "assistant", "content": full_target}]
            full_text = text_tokenizer.apply_chat_template(
                full_msgs,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception:
            continue

        token_len = len(text_tokenizer.encode(full_text, add_special_tokens=False))
        if 40 < token_len <= TARGET_SEQ_LEN:
            total_tokens += token_len
            pbar.set_postfix({"Tokens": f"{total_tokens / 1e6:.2f}M"})
            yield {
                "prompt": prompt_str,
                "text": full_text,
                "ground_truth": record["ground_truth"],
                "task_type": record["task_type"],
            }
            pbar.update(1)

    pbar.close()

print("Building alignment dataset...")
raw_alignment_dataset = Dataset.from_generator(alignment_generator)
split_dataset = raw_alignment_dataset.train_test_split(test_size=2500, seed=3407, shuffle=True)

print(f"Serializing cache to {LOCAL_ALIGNMENT_CACHE_DIR}...")
split_dataset.save_to_disk(LOCAL_ALIGNMENT_CACHE_DIR)

print("\n✓ Alignment Cache Created Successfully!")
print(f"Total Train Samples: {len(split_dataset['train']):,}")
print(f"Total Eval Samples:  {len(split_dataset['test']):,}")
