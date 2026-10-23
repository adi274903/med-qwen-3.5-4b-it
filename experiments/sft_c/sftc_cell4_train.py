import gc, json, math, os, re, shutil, time
import torch
from datasets import load_from_disk
from transformers import TrainerCallback
from tqdm.auto import tqdm
from unsloth import UnslothTrainer, UnslothTrainingArguments
from unsloth.chat_templates import train_on_responses_only

gc.collect(); torch.cuda.empty_cache()

# ---- pre-flight: Qwen3.5's linear-attention layers need flash-linear-attention (fla). Without it transformers silently falls back to
#      a pure-PyTorch path whose backward pass is extremely slow (training sits at step 0). Install it, then RESTART the session.
import importlib.util, sys
assert importlib.util.find_spec("fla") is not None, \
    "flash-linear-attention is not installed: run  pip install -U flash-linear-attention  then Runtime -> Restart session"
_fast = {k: bool(v.is_fast_path_available) for k, v in sys.modules.items() if "qwen3_5" in k and hasattr(v, "is_fast_path_available")}
print("fla installed | transformers fast-path flag:", _fast or "unknown")   # the flag can read False when only causal-conv1d is missing: that is OK

BASELINE_EVAL = False       # the pre-SFT baseline on this exact split/model was already measured: 1.3084 (PPL 3.70); True re-measures it (~70 s)

DRIVE_CHECKPOINTS_DIR = os.path.join(DRIVE_ROOT, "checkpoints")
DRIVE_TENSORBOARD_DIR = os.path.join(DRIVE_ROOT, "tensorboard_logs")
DRIVE_FINAL_DIR = os.path.join(DRIVE_ROOT, "final_adapter")
for d in (DRIVE_CHECKPOINTS_DIR, DRIVE_TENSORBOARD_DIR, DRIVE_FINAL_DIR):
    os.makedirs(d, exist_ok=True)
os.environ["TENSORBOARD_LOGGING_DIR"] = DRIVE_TENSORBOARD_DIR

sft_data = load_from_disk(os.path.join(CACHE_DIR, "split"))
train_dataset, eval_dataset = sft_data["train"], sft_data["test"]
print(f"Train {len(train_dataset):,} | Eval {len(eval_dataset):,}")

eff_batch = BATCH_SIZE * GRAD_ACCUM
steps = min(TOTAL_STEPS, len(train_dataset) // eff_batch)
if steps < TOTAL_STEPS:
    print(f"[note] dataset only supports {steps} steps of {eff_batch}; max_steps reduced from {TOTAL_STEPS} to {steps}")


class PerplexityLoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            ppl = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 50 else float("inf")
            metrics["eval_perplexity"] = ppl
            print(f"\n>>> [Step {state.global_step}] Eval CE Loss: {metrics['eval_loss']:.4f} | Eval Perplexity: {ppl:.2f}", flush=True)


class SFTProgressBar(TrainerCallback):
    """One live tqdm bar (steps, elapsed, ETA, s/step, last loss, peak memory) plus a history line every 50 steps."""
    def __init__(self):
        self.bar, self.t0, self.start_step, self.last_loss = None, None, 0, "NA"

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.perf_counter()
        self.start_step = state.global_step                 # non-zero when resuming
        self.bar = tqdm(total=state.max_steps, initial=state.global_step, desc="SFT", unit="step",
                        dynamic_ncols=True, smoothing=0.05, leave=True)
        tqdm.write(f"[train] started at step {self.start_step}")

    def on_step_end(self, args, state, control, **kwargs):
        if self.bar is None:
            return
        if state.global_step > self.bar.n:
            self.bar.update(state.global_step - self.bar.n)
        self.bar.set_postfix(loss=self.last_loss, peak=f"{torch.cuda.max_memory_allocated() / 2**30:.1f}G", refresh=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.bar is None:                                # on_log can fire during evaluate(), before training starts
            return
        if logs and "loss" in logs:
            self.last_loss = f"{logs['loss']:.3f}"
        done = state.global_step - self.start_step
        if done and state.global_step % 50 == 0:
            el = time.perf_counter() - self.t0
            sps = el / done
            eta = sps * (state.max_steps - state.global_step)
            tqdm.write(f"Step {state.global_step}/{state.max_steps} | {el / 60:.0f} min | ETA {eta / 3600:.2f} h | "
                       f"{sps:.2f} s/step | loss {self.last_loss}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.bar is not None:
            self.bar.close()


training_args = UnslothTrainingArguments(
    output_dir=DRIVE_CHECKPOINTS_DIR,
    run_name=RUN_NAME,
    report_to="tensorboard",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    per_device_eval_batch_size=BATCH_SIZE,   # 16 was slower (1:56 vs 1:10): bigger batches pad more
    max_length=MAX_SEQ_LENGTH,
    packing=False,
    # to test length grouping use train_sampling_strategy="group_by_length" (this transformers version ignores group_by_length)
    disable_tqdm=True,                       # our bar below replaces the notebook widget that did not render
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=steps,
    optim="adamw_torch_fused",
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=1.0,
    weight_decay=0.01,
    bf16=True,
    fp16=False,
    gradient_checkpointing=GRAD_CKPT,
    dataloader_num_workers=0,                # 2 workers can hang when forked after the threaded data build in the same process
    dataloader_pin_memory=True,
    save_strategy="steps",
    save_steps=EVAL_STEPS,
    save_total_limit=2,
    seed=SEED,
)

trainer = UnslothTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    callbacks=[PerplexityLoggingCallback(), SFTProgressBar()],
)
trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")

# ---- auto-resume from the newest checkpoint on Drive (needs the same cache, the same settings, and cells 1, 3, 4 re-run)
ckpts = [d for d in os.listdir(DRIVE_CHECKPOINTS_DIR) if re.fullmatch(r"checkpoint-\d+", d)]
resume_path = os.path.join(DRIVE_CHECKPOINTS_DIR, max(ckpts, key=lambda d: int(d.split("-")[1]))) if ckpts else None
print("Resuming from:", resume_path or "scratch")

if resume_path is None and BASELINE_EVAL:    # baseline eval only on a fresh run
    print("Computing the pre-SFT baseline eval loss...")
    base_loss = trainer.evaluate()["eval_loss"]
    print(f"Pre-SFT baseline CE loss: {base_loss:.4f} | PPL {math.exp(base_loss):.2f}")
    # SFT A on the DAPT-merged model started at 1.2597 (its own held-out split, so not exactly comparable).

trainer.train(resume_from_checkpoint=resume_path)

print(f"Training complete. Saving the adapter to {DRIVE_FINAL_DIR} ...")
model.save_pretrained(DRIVE_FINAL_DIR)
text_tokenizer.save_pretrained(DRIVE_FINAL_DIR)

# provenance: keep the exact data stats and run settings next to the weights
shutil.copy(os.path.join(CACHE_DIR, "stats.json"), os.path.join(DRIVE_ROOT, "data_stats.json"))
json.dump(dict(run=RUN_NAME, base=BASE_MODEL_ID, use_dapt=USE_DAPT, steps=steps, eff_batch=eff_batch, lr=LEARNING_RATE,
               seed=SEED, grad_ckpt=GRAD_CKPT, batch_size=BATCH_SIZE, grad_accum=GRAD_ACCUM, lora_r=128, lora_alpha=256),
          open(os.path.join(DRIVE_ROOT, "run_config.json"), "w"), indent=2)
print("Saved.")
