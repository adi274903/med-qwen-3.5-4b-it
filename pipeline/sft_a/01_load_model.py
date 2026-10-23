import torch
from peft import PeftModel
from unsloth import FastModel

MODEL_ID = "Qwen/Qwen3.5-4B-Base"
PREV_LORA_REPO = "adi274903/Med-Qwen-3.5-4B-Base-DAPT-Adapter"
MAX_SEQ_LENGTH = 2048

# 1. Load base model in native BF16
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    offload_embedding=False,
)

# 2. Attach previous LoRA adapter and merge it into base weights
model = PeftModel.from_pretrained(model, PREV_LORA_REPO)
model = model.merge_and_unload()

# Ensure model is back in training mode and prepare tokenizer
model.train()
text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
text_tokenizer.padding_side = "right"
if text_tokenizer.pad_token_id is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token

model.config.use_cache = False

# 3. Apply your new LoRA adapter configuration on the merged model
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    modules_to_save=None,
    r=128,
    lora_alpha=256,
    lora_dropout=0.0,
    bias="none",
    use_dora=False,
    use_gradient_checkpointing=False,
    random_state=3407,
)

model.print_trainable_parameters()
