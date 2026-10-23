# Diagnose why RLVR-C-noDAPT has only 62% format-valid replies on PubMedQA (generative eval).
# Run in a runtime where /content/merged_stage_models/RLVR-C-noDAPT exists (left behind by the eval cell),
# or after re-running the eval cell for RLVR-C-noDAPT. Self-contained: prints raw failing outputs.
import collections, os
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_PATH = "/content/merged_stage_models/RLVR-C-noDAPT"
assert os.path.isdir(MODEL_PATH), "merged model not found: run the eval cell first (it merges on local disk)"
N = 200
INSTR = '\n\nBased on the context, think step by step, then finish with "The answer is X." where X is yes, no, or maybe.'

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base")
rows = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train").select(range(N))
def chat_prompt(u):
    p = tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    return p if p.rstrip().endswith("<think>") else p + "<think>\n"
prompts = [chat_prompt(f"Context: {' '.join(r['context']['contexts'])}\nQuestion: {r['question']}" + INSTR) for r in rows]
print("prompt tail:", repr(prompts[0][-60:]))

llm = LLM(model=MODEL_PATH, tokenizer=MODEL_PATH, dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=4096, seed=3407)
outs = llm.generate(prompts, SamplingParams(max_tokens=1536, temperature=0.0))

kinds, bad = collections.Counter(), []
for o in outs:
    t = o.outputs[0].text
    n_close, n_open = t.count("</think>"), t.count("<think>")
    if n_close == 1 and n_open <= 1:
        kinds["valid"] += 1
    else:
        kinds[f"close={n_close},open={n_open}"] += 1
        bad.append(t)
print("\nformat breakdown:", dict(kinds))
for t in bad[:6]:
    print("\n--- invalid reply ---\n" + repr(t[-500:]))
