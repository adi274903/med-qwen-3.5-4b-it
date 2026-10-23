# ==========================================
# CELL 1 (plain TRL, no Unsloth) -- run in a FRESH runtime.
# 1) Install (uncomment once; vLLM pins its own torch, so restart the runtime after installing):
#      %pip install -U "trl[vllm]" peft accelerate datasets flash-linear-attention
#    Do NOT install/import unsloth in this runtime.
# 2) Builds the RLVR starting model: Base + DAPT + SFT A merged and saved to Drive.
# ==========================================
import gc
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID = "Qwen/Qwen3.5-4B-Base"
DAPT_ID = "adi274903/Med-Qwen-3.5-4B-Base-DAPT-Adapter"
SFTA_ID = "adi274903/Med-Qwen-3.5-4B-SFT-Adapter"
MODEL_DIR = "/content/drive/MyDrive/Med_Qwen_4B_DAPT_SFTA_merged"

if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    print("Merged checkpoint already exists:", MODEL_DIR)
else:
    m = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.bfloat16, device_map="auto")
    m = PeftModel.from_pretrained(m, DAPT_ID).merge_and_unload()    # DAPT first,
    m = PeftModel.from_pretrained(m, SFTA_ID).merge_and_unload()    # then SFT A
    m.save_pretrained(MODEL_DIR)
    AutoTokenizer.from_pretrained(BASE_ID).save_pretrained(MODEL_DIR)
    del m
    gc.collect()
    torch.cuda.empty_cache()
    print("Saved merged checkpoint to", MODEL_DIR)
