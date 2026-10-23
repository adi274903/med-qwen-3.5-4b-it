import os
import gc
from datasets import load_dataset, interleave_datasets, Dataset, DatasetDict
from tqdm.auto import tqdm

# ==========================================
# 1. Pipeline Constants
# ==========================================
TARGET_SEQ_LEN = 2048
LOCAL_SFT_CACHE_DIR = "/content/sft_medical_reasoning_cache"
os.makedirs(LOCAL_SFT_CACHE_DIR, exist_ok=True)

print("Connecting to curated SFT streams (Medical CoT + Qwen Reasoning + Multi-turn)...")

# 1. Medical Long-CoT Reasoning
med_o1_stream = load_dataset(
    "FreedomIntelligence/medical-o1-reasoning-SFT",
    "en",
    split="train",
    streaming=True
).take(45_000)

ii_med_stream = load_dataset(
    "Intelligent-Internet/II-Medical-Reasoning-SFT",
    split="train",
    streaming=True
).take(50_000)

# 2. Direct Medical Exam QA & Clinical Instruction
medqa_stream = load_dataset(
    "medalpaca/medical_meadow_medqa",
    split="train",
    streaming=True
)

alpacare_stream = load_dataset(
    "lavita/AlpaCare-MedInstruct-52k",
    split="train",
    streaming=True
)

# 3. General Instruction Anchor & Quantitative Anti-Forgetting
ultrachat_stream = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split="train_sft",
    streaming=True
).take(25_000)

orca_math_stream = load_dataset(
    "microsoft/orca-math-word-problems-200k",
    split="train",
    streaming=True
).take(20_000)

# ==========================================
# 2. Standardized Chat Extraction Formatters
# ==========================================
def clean_string(val):
    return val.strip() if isinstance(val, str) else ""

def format_med_o1(ex):
    q = clean_string(ex.get("Question"))
    cot = clean_string(ex.get("Complex_CoT"))
    res = clean_string(ex.get("Response"))

    if cot:
        assistant_content = f"<think>\n{cot}\n</think>\n\n{res}"
    else:
        assistant_content = res

    return {
        "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": assistant_content}
        ]
    }

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
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    }

def format_medqa(ex):
    prompt = clean_string(ex.get("input"))
    response = clean_string(ex.get("output"))
    return {
        "messages": [
            {"role": "user", "content": f"Clinical Question:\n{prompt}"},
            {"role": "assistant", "content": response}
        ]
    }

def format_alpacare(ex):
    inst = clean_string(ex.get("instruction"))
    inp = clean_string(ex.get("input"))
    out = clean_string(ex.get("output"))
    prompt = f"{inst}\n\nCase Details: {inp}" if inp else inst
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": out}
        ]
    }

def format_ultrachat(ex):
    raw_msgs = ex.get("messages", [])
    messages = [
        {"role": m.get("role", "user"), "content": clean_string(m.get("content"))}
        for m in raw_msgs if m.get("content")
    ]
    return {"messages": messages}

def format_orca(ex):
    q = clean_string(ex.get("question"))
    a = clean_string(ex.get("answer"))
    return {
        "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}
        ]
    }

def prepare_sft_stream(ds, formatter):
    cols = list(ds.features.keys()) if ds.features else list(next(iter(ds)).keys())
    return (
        ds.map(formatter, remove_columns=cols)
          .filter(lambda x: len(x.get("messages", [])) >= 2 and len(x["messages"][-1]["content"]) > 10)
    )

# 6-Part SFT Stream: 55% Clinical CoT/QA, 25% Board Exam & Case Studies, 20% Math/Chat Anchors
streams = [
    prepare_sft_stream(med_o1_stream, format_med_o1),        # 28%
    prepare_sft_stream(ii_med_stream, format_ii_med),        # 27%
    prepare_sft_stream(medqa_stream, format_medqa),          # 15%
    prepare_sft_stream(alpacare_stream, format_alpacare),    # 10%
    prepare_sft_stream(ultrachat_stream, format_ultrachat),  # 10%
    prepare_sft_stream(orca_math_stream, format_orca),       # 10%
]

probs = [0.28, 0.27, 0.15, 0.10, 0.10, 0.10]

interleaved_sft_stream = interleave_datasets(
    streams,
    probabilities=probs,
    seed=3407,
    stopping_strategy="all_exhausted"
)

# ==========================================
# 3. Apply Chat Template, Filter & Train/Eval Split
# ==========================================
def sft_generator():
    pbar = tqdm(desc="Processing SFT Conversations", unit="samples")

    for record in interleaved_sft_stream:
        messages = record["messages"]
        try:
            formatted_text = text_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception:
            continue

        token_len = len(text_tokenizer.encode(formatted_text, add_special_tokens=False))
        if 50 < token_len <= TARGET_SEQ_LEN:
            yield {"text": formatted_text}
            pbar.update(1)

    pbar.close()

print("Extracting and building combined dataset...")
raw_sft_dataset = Dataset.from_generator(sft_generator)

print("Creating Train / Eval splits (holding out 2,000 for validation)...")
split_dataset = raw_sft_dataset.train_test_split(test_size=2000, seed=3407, shuffle=True)

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"Serializing SFT train/eval dataset to {LOCAL_SFT_CACHE_DIR}...")
split_dataset.save_to_disk(LOCAL_SFT_CACHE_DIR)

print("\nSFT Local Cache Created Successfully!")
print(f"Total Train Samples: {len(train_dataset):,}")
print(f"Total Eval Samples:  {len(eval_dataset):,}")
print(f"Saved at: {LOCAL_SFT_CACHE_DIR}")
