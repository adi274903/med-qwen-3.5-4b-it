import os, shutil
from safetensors.torch import load_file, save_file

SRC = os.path.join(DRIVE_ROOT, "final_adapter")
DST = os.path.join(DRIVE_ROOT, "final_adapter_peft_fixed")
os.makedirs(DST, exist_ok=True)
for f in os.listdir(SRC):
    p = os.path.join(SRC, f)
    if os.path.isfile(p) and not f.endswith((".safetensors", ".bin", ".pt")):
        shutil.copy2(p, DST)

# Unsloth saves  base_model.model.model.language_model.layers.N...  (multimodal wrapper names);
# the eval / RLVR loaders use AutoModelForCausalLM  ->  base_model.model.model.layers.N...  (same as the working SFT A adapter)
tensors = load_file(os.path.join(SRC, "adapter_model.safetensors"))
def fix(k):
    k = k.replace("base_model.model.model.model.", "base_model.model.model.", 1)   # doubled prefix, if present
    return k.replace(".language_model.", ".", 1)
fixed = {fix(k): v for k, v in tensors.items()}
assert len(fixed) == len(tensors), "key collision while remapping"
print(f"remapped {sum(fix(k) != k for k in tensors)} of {len(tensors)} keys")
print("before:", sorted(tensors)[0])
print("after :", sorted(fixed)[0])
save_file(fixed, os.path.join(DST, "adapter_model.safetensors"), metadata={"format": "pt"})
print("Saved:", DST)

# ---- verify: load it the way the eval does (CPU, AutoModelForCausalLM + PeftModel) and make sure EVERY lora_B is non-zero.
#      A zero lora_B means the keys did not match and the adapter would be a silent no-op.
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
chk = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
chk = PeftModel.from_pretrained(chk, DST, is_trainable=False)
b = {n: p.float().abs().sum().item() for n, p in chk.named_parameters() if "lora_B" in n}
zero = [n for n, v in b.items() if v == 0]
print(f"lora_B tensors: {len(b)} | all-zero: {len(zero)}")
assert len(b) == 128 and not zero, f"adapter did not load correctly, e.g. {zero[:2]}"
print("OK: adapter keys match the model; the weights loaded.")
del chk; import gc; gc.collect()
