import os
import gc
import math
import torch
from datasets import load_from_disk
from transformers import TrainerCallback
from unsloth import UnslothTrainer, UnslothTrainingArguments
from unsloth.chat_templates import train_on_responses_only

# ==========================================
# 1. Clear GPU Memory State
# ==========================================
gc.collect()
torch.cuda.empty_cache()

# ==========================================
# 2. Checkpoint & Logging Paths (Google Drive)
# ==========================================
DRIVE_ROOT = "/content/drive/MyDrive/Med_Qwen_4B_SFT"
DRIVE_CHECKPOINTS_DIR = os.path.join(DRIVE_ROOT, "checkpoints")
DRIVE_TENSORBOARD_DIR = os.path.join(DRIVE_ROOT, "tensorboard_logs")
DRIVE_FINAL_DIR       = os.path.join(DRIVE_ROOT, "final_adapter")

os.makedirs(DRIVE_CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(DRIVE_TENSORBOARD_DIR, exist_ok=True)
os.makedirs(DRIVE_FINAL_DIR, exist_ok=True)

# ==========================================
# 3. Load SFT Local Cache (Already Pre-Split)
# ==========================================
CACHE_PATH = "/content/sft_medical_reasoning_cache"
assert os.path.exists(CACHE_PATH), f"SFT Cache not found at {CACHE_PATH}!"

sft_data = load_from_disk(CACHE_PATH)
train_dataset = sft_data["train"]
eval_dataset  = sft_data["test"]

print(f"Loaded SFT Cache successfully.")
print(f"Train split: {len(train_dataset):,} conversations | Eval split: {len(eval_dataset):,} conversations")

# ==========================================
# 4. Callback: Compute & Push PPL to TensorBoard
# ==========================================
class PerplexityLoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            try:
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")
            metrics["eval_perplexity"] = perplexity
            print(f"\n>>> [Step {state.global_step}] Eval CE Loss: {eval_loss:.4f} | Eval Perplexity: {perplexity:.2f}")

# ==========================================
# 5. Training Configuration with Eval & TensorBoard
# ==========================================
TARGET_SEQ_LEN = 2048
BATCH_SIZE = 4
GRAD_ACCUM = 16  # Effective batch = 64 conversations (~131k tokens max)
TOTAL_STEPS = 2700

training_args = UnslothTrainingArguments(
    output_dir=DRIVE_CHECKPOINTS_DIR,
    run_name="med_qwen_4b_sft_reasoning_run",

    # TensorBoard setup
    report_to="tensorboard",
    logging_dir=DRIVE_TENSORBOARD_DIR,
    logging_steps=10,

    # Evaluation setup (runs every 150 steps)
    eval_strategy="steps",
    eval_steps=150,
    per_device_eval_batch_size=BATCH_SIZE,

    # Sequence Length & Batch Budget
    max_length=TARGET_SEQ_LEN,
    packing=False,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=TOTAL_STEPS,

    # SFT Optimization
    optim="adamw_torch_fused",
    learning_rate=2e-5,          # Lowered from 1e-4 for stable SFT
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=1.0,
    weight_decay=0.01,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,

    # Dataloader configuration
    dataloader_num_workers=2,
    dataloader_pin_memory=True,

    # Checkpoint Persistence
    save_strategy="steps",
    save_steps=150,
    save_total_limit=2,
    seed=3407,
)

trainer = UnslothTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=TARGET_SEQ_LEN,
    callbacks=[PerplexityLoggingCallback()],
)

# Enforce response-only loss masking for Qwen ChatML markers
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# Correct token budget calculation
tokens_per_step = (
    training_args.per_device_train_batch_size
    * training_args.gradient_accumulation_steps
    * TARGET_SEQ_LEN
)
print(f"Max throughput budget: {tokens_per_step:,} tokens per optimizer step.")

if "SFTThroughputMonitor" in globals():
    trainer.add_callback(SFTThroughputMonitor(training_args.max_steps, tokens_per_step))
elif "DAPTThroughputMonitor" in globals():
    trainer.add_callback(DAPTThroughputMonitor(training_args.max_steps, tokens_per_step))

# ==========================================
# 6. Launch Training & Save Adapter
# ==========================================
print("Computing initial baseline eval loss on held-out SFT split...")
baseline_metrics = trainer.evaluate()
baseline_loss = baseline_metrics["eval_loss"]
print(f"Pre-SFT Baseline CE Loss: {baseline_loss:.4f} | Baseline PPL: {math.exp(baseline_loss):.2f}\n")

trainer.train(resume_from_checkpoint=False)

print(f"Training complete. Persisting final SFT adapter to {DRIVE_FINAL_DIR}...")
model.save_pretrained(DRIVE_FINAL_DIR)
text_tokenizer.save_pretrained(DRIVE_FINAL_DIR)
print("Artifacts successfully saved to Google Drive.")
