"""
QLoRA Fine-tuning for Llama 3.2 3B (local HF weights)
-------------------------------------------------------
NOTE: This script requires the Llama 3.2 3B weights in HuggingFace format.
      Ollama stores weights in its own format and CANNOT be used for training.

      To get HF weights:
        Option A — Download from HuggingFace (requires internet once):
          pip install huggingface_hub
          huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
              --local-dir ./models/llama3.2-3b

        Option B — Convert from Ollama blobs (advanced, not recommended).

      After download, set BASE_MODEL_PATH below and run this script offline.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

os.environ["WANDB_DISABLED"]        = "true"
os.environ["TRANSFORMERS_OFFLINE"]  = "1"   # remove this line for first-time download
os.environ["HF_HUB_OFFLINE"]        = "1"   # remove this line for first-time download

# ---------------- PATHS ----------------
BASE_MODEL_PATH = r"models\llama3.2-3b"   # <-- HuggingFace format weights folder
DATASET_PATH    = r"finetuning\data\qa_pairs_llama.jsonl"
OUTPUT_DIR      = r"models\llama3.2_qlora_adapters"

# ---------------- CHECK CUDA ----------------
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ CUDA not detected. Training will be very slow on CPU.")

# ---------------- QUANTIZATION CONFIG ----------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# ---------------- LOAD TOKENIZER ----------------
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
)

# Llama 3.2 uses <|eot_id|> as EOS — set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ---------------- LOAD MODEL ----------------
print("\nLoading Llama 3.2 3B in 4-bit (QLoRA)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
)
model.config.use_cache = False
print("✅ Model loaded")

# ---------------- LORA CONFIG ----------------
print("\nSetting up LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Llama 3.2 attention + MLP projection names
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# ---------------- LOAD DATASET ----------------
print("\nLoading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = dataset["train"]
eval_dataset  = dataset["test"]

print(f"Train samples : {len(train_dataset)}")
print(f"Eval  samples : {len(eval_dataset)}")

# ---------------- FORMAT DATA ----------------
# Expected JSONL format — each line must have a "messages" field, e.g.:
# {
#   "messages": [
#     {"role": "system",    "content": "You are a helpful assistant."},
#     {"role": "user",      "content": "What is X?"},
#     {"role": "assistant", "content": "X is ..."}
#   ]
# }

def format_example(example):
    """Apply Llama 3.2 chat template to each example."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

train_dataset = train_dataset.map(format_example)
eval_dataset  = eval_dataset.map(format_example)

# Quick sanity check on formatted output
print("\nSample formatted text (first 300 chars):")
print(train_dataset[0]["text"][:300])
print("...")

# ---------------- TRAINING ARGUMENTS ----------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,    # safe for GTX 1650 / 4 GB VRAM
    gradient_accumulation_steps=8,    # effective batch = 8
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="epoch",            # updated arg name (evaluation_strategy deprecated)
    save_total_limit=2,
    report_to="none",
    optim="paged_adamw_8bit",
)

# ---------------- TRAINER ----------------
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=1024,
    args=training_args,
)

# ---------------- TRAIN ----------------
print("\n🚀 Starting QLoRA Fine-tuning on Llama 3.2 3B...\n")
trainer.train()

# ---------------- SAVE ----------------
print("\n💾 Saving LoRA adapters + tokenizer...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ Fine-tuning complete!")
print(f"Adapters saved at: {OUTPUT_DIR}")
print(
    "\nTo use these adapters in inference.py, swap OLLAMA_MODEL for PeftModel loading.\n"
    "See the README or ask for a merged-adapter inference script."
)