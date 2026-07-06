import os
import torch
import gc
import sys

# VRAM CONTROL + LEGACY HARDWARE BOUNDARIES:
# enforcing strict hardware allocation ceiling for the training process.
# L4 has 24GB, 0.58 of 24GB = ~13.9 GB
# Combined with vLLM's 30% (7.2GB), total usage = under 90% cap, ~21.1 GB  ceiling
torch.cuda.set_per_process_memory_fraction(0.58, 0)

# Disable custom memory allocation engines that fight over PyTorch's native cache allocator
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datasets import Dataset, load_from_disk, load_dataset
# using hugging face libraries...
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from transformers.trainer_callback import TrainerCallback

# custom callback to clean GPU cache immediately after each training step
class MemoryGarbageCollectorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()

# PATH + CONFIG SETTINGS
MODEL_ID = "Qwen/Qwen2.5-7B"
OUTPUT_DIR = "./qwen-bio-summarizer-lora"
DATASET_PATH = "./data"

# TOKENIZATION + BIOLOGY PROMPT TEMPLATE MAPPING
print("Loading Model Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# automatically scan, import local hugging face dataset folder structure
if not os.path.exists(DATASET_PATH):
    print(f"Error: The directory '{DATASET_PATH}' does not exist.", file=sys.stderr)
    print("Please update DATASET_PATH to point to your local dataset folder.", file=sys.stderr)
    sys.exit(1)

print("Loading native Arrow dataset from disk...")
try:
    #arrow format datasets from HF unpacked via load_from_disk
    raw_dataset = load_from_disk(DATASET_PATH)
except Exception:
    #fallback if is a directory w/ individual arrow parts
    raw_dataset = load_dataset("arrow", data_dir=DATASET_PATH)

# extract active split (handle both DatasetDict container structs or direct dataset formats)
if isinstance(raw_dataset, dict) or hasattr(raw_dataset, "keys"):
    dataset_split = raw_dataset["train"] if "train" in raw_dataset else next(iter(raw_dataset.values()))
else:
    dataset_split = raw_dataset

# print actual columns present in Arrow structure
print (f"Detected dataset columns: {dataset_split.column_names}")

# dynamically locate text, summary keys out of standard variations
TEXT_KEYS = ["text", "article", "context", "content", "document", "inputs"]
SUMMARY_KEYS = ["summary", "abstract", "targets", "output"]

found_text_key = next((k for k in TEXT_KEYS if k in dataset_split.column_names), None)
found_summary_key = next((k for k in SUMMARY_KEYS if k in dataset_split.column_names), None)

if not found_text_key or not found_summary_key:
    print(f"Error: Could not automatically map biology fields.", file=sys.stderr)
    print(f"Available fields: {dataset_split.column_names}", file=sys.stderr)
    print("Please explicitly match them in the script.", file=sys.stderr)
    sys.exit(1)

print(f"Mapping inputs from column: '{found_text_key}' | Summaries from column: '{found_summary_key}'")

# for ARROW: filter out corrupted/null/empty rows before batch mapping 
#def remove_empty_arrow_rows(example):
    #dynamically find right columns in arrow table schema 
#    art = example.get("text") or example.get("article")
#    summ = example.get("summary") or example.get("abstract")

  #  if art is None or summ is None:
   #     return False
   # if str(art).strip() == "" or str(summ).strip() == "":
   #     return False
   # return True

#print(f"Scanning arrow tables for empty rows.. Original size: {len(dataset_split)}")
#dataset_split = dataset_split.filter(remove_empty_arrow_rows)
#print(f"Cleaned dataset ready. Stable rows remaining: {len(dataset_split)}")

def format_bio_prompt(examples):
    # Dynamically handle standard HF text fields (text/article & summary/abstract)
    articles = examples.get(found_text_key, [])
    summaries = examples.get(found_summary_key, [])

    inputs = []
    targets = []
    for art, summ in zip(articles, summaries):
        if art is None or summ is None:
            continue
        art_str = str(art).strip()
        summ_str = str(summ).strip()
        if len(art_str) == 0 or len(summ_str) == 0:
            continue

        prompt = (
            f"<|im_start|>system\nYou are an expert biologist. "
            f"Summarize the following article accurately, preserving technical terms, "
            f"methodologies, and key discoveries, but making it readable for a lay audience.<|im_end|>\n"
            f"<|im_start|>user\n{art_str}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs.append(prompt)
        targets.append(summ_str + tokenizer.eos_token)

    #return raw text pairs to map cleanly
    return {"prompt_text": inputs, "target_text": targets}

print("formatting dataset text frames...")
mapped_dataset = dataset_split.map(format_bio_prompt, batched=True, remove_columns=dataset_split.column_names)

# tokenize processed texts into explicit tensor structures 
def tokenize_pairs(examples):
    model_inputs = tokenizer(examples["prompt_text"], max_length=1536, truncation=True)
    labels = tokenizer(text_target=examples["target_text"], max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing and building prompt arrays...")
tokenized_dataset = mapped_dataset.map(tokenize_pairs, batched=True, remove_columns=mapped_dataset.column_names)
print(f"Final training dataset ready. stable token rows: {len(tokenized_dataset)}")

if len(tokenized_datset) == 0:
    print("Error: tokenized dataset has 0 samples. verify sample row contents.", file=sys.stderr)
    sys.exit(1) 

# optimized parameter-efficient LoRA Settings
lora_config =  LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], # Reduced targets slightly to save tensor memory
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

print("📥 Loading base model weights in FP16 format...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16, #would otherwise use BF16, but using FP16 layout here for safety w/ legacy architecture
    device_map="auto"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Memory-Restricted Training ARGUMENTS + EXECUTION
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,     # Hard minimum input allocation
    gradient_accumulation_steps=16,    # High accumulation to keep effective batch size stable at 16 without VRAM scaling
    learning_rate=1e-4,
    num_train_epochs=2,
    logging_steps=5,
    fp16=True,                         # Enforces standard half precision config loops
    bf16=False,                         # Explicitly turn off to guarantee safety w/ legacy arch
    gradient_checkpointing=True,       # CRITICAL: Frees internal layer activations during backward passes
    save_strategy="epoch",
    report_to="none",
    remove_unused_columns=False        # stop signature matching validation error 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    callbacks=[MemoryGarbageCollectorCallback()] # Binds our custom VRAM cleanup rule
)

if __name__ == "__main__":
    print("aunching m90% memory-capped, FP16 Biology fine-tuning execution window...")
    try:
        trainer.train()
        model.save_pretrained(OUTPUT_DIR)
        print(f"\nSuccess! Trained legacy-compliant LoRA weights saved to: {OUTPUT_DIR}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\nOut of Memory Error. Please ensure background vLLM instance is capped correctly.")
        else:
            raise e


