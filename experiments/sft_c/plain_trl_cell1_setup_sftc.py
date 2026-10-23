# ==========================================
# CELL 1 (plain TRL, no Unsloth), SFT C (no DAPT) -- run in a FRESH runtime.
# 1) Install (uncomment once; vLLM pins its own torch, so restart the runtime after installing):
#      %pip install -U "trl[vllm]" peft accelerate datasets flash-linear-attention
#    Do NOT install/import unsloth in this runtime.
# 2) Builds Base + SFT C and saves it to a NEW Drive folder.
# Base + SFT C (trained on the plain base, so no adapter/base mismatch). Evaluate it before spending time on RLVR.
# ==========================================
import gc
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID = "Qwen/Qwen3.5-4B-Base"
SFTC_ID = "/content/drive/MyDrive/Med_Qwen_4B_SFT_C_noDAPT/final_adapter_peft_fixed"   # SFT C
MODEL_DIR = "/content/drive/MyDrive/Med_Qwen_4B_BASE_SFTC_noDAPT_merged"   # new folder, different from the DAPT merge

if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    print("Merged checkpoint already exists:", MODEL_DIR)
else:
    m = AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=torch.bfloat16, device_map="auto")
    m = PeftModel.from_pretrained(m, SFTC_ID).merge_and_unload()    # SFT C directly on the base (no DAPT)
    m.save_pretrained(MODEL_DIR)
    AutoTokenizer.from_pretrained(BASE_ID).save_pretrained(MODEL_DIR)
    del m
    gc.collect()
    torch.cuda.empty_cache()
    print("Saved merged checkpoint to", MODEL_DIR)

