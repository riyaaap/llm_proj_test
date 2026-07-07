import os
import sys

# forces PyTorch import to only see ONE specific GPU by ID/index name from nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# limit VRAM use to stay safe under ceiling
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import torch # import AFTER setting env variable to select specific GPU to use

# limiting VRAM use to 13.9 GB  w/ vLLM capped at 30% to stay under 21.6GB 90% total use ceiling
torch.cuda.set_per_process_memory_fraction(0.72, 0)
# run fails with CUDA out of memory error if use 0.58

# training script for text summarization, trained on dataset found at "ambrosfitz/cnn-daily-grammar" on Hugging Face:

from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers.trainer_callback import TrainerCallback

class MemoryGarbageCollectorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()

# path and config settings
MODEL_ID = "Qwen/Qwen2.5-7B"
STAGE1_OUTPUT_DIR = "./qwen-stage1-cnn-lora"
MERGED_OUTPUT_DIR = "./Qwen2.5-7B-Summarizer"
CNN_DATASET_PATH = "./cnn_data"

#tokenisation, supervised task prompt mapping

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

if not os.path.exists(CNN_DATASET_PATH):
    print(f"Error: The directory '{CNN_DATASET_PATH}' does not exist.", file=sys.stderr)
    sys.exit(1)

print("Loading Arrow CNN article dataset from disk...")
try:
    raw_dataset = load_from_disk(CNN_DATASET_PATH)
except Exception:
    raw_dataset = load_dataset("arrow", data_dir=CNN_DATASET_PATH)

if isinstance(raw_dataset, dict) or hasattr(raw_dataset, "keys"):
    dataset_split = raw_dataset["train"] if "train" in raw_dataset else next(iter(raw_dataset.values()))
else:
    dataset_split = raw_dataset

print(f"Detected dataset columns: {dataset_split.column_names}")

# single-row transformation to avoid array dimension mismatch issues

def process_single_row(example):
    art = example.get("article")
    summ = example.get("summary")

    # Skip invalid rows safely by returning empty lists for the trainer to skip

    if not art or not summ or len(str(art).strip()) == 0 or len(str(summ).strip()) == 0:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    prompt = (
        f"<|im_start|>system\nYou are an expert assistant. "
        f"Summarize the following article accurately, preserving technical terms, "
        f"methods, and key focuses, in an easibly readable manner that lay audiences can understand.<|im_end|>\n"
        f"<|im_start|>user\n{str(art).strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    target_text = f"{str(summ).strip()}{tokenizer.eos_token}"

    # Tokenize both blocks separately to calc lengths precisely
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_tokens = tokenizer(target_text, add_special_tokens=False)["input_ids"]

    # combine into one single continuous string array
    input_ids = prompt_tokens + target_tokens

    # mask prompt part w -100 so model only calcs loss on generating summary
    labels = [-100] * len(prompt_tokens) + target_tokens

    # enforce max length boundaries
    # was initially 2048 then decreased to -> 1024 --> 512 to lower gpu memory use
    if len(input_ids) > 512:
        input_ids = input_ids[:512]
        labels = labels[:512]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


print("Tokenizing dataset into unified causal map")
#process row-by-row by removing batched=True
tokenized_dataset = dataset_split.map(
    process_single_row,
    batched=False,
    remove_columns=dataset_split.column_names
)

# clean out any caught empty/skipped rows
tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)
print(f"Stage 1 training dataset pairs balanced. Rows: {len(tokenized_dataset)}")

# parameter-efficient LoRA setup:
# initially had rank r=16 and alpha=32, but dropped to compress adapter memory, drop gradient state size
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    # for target blocks initially had "gate_proj", "up_proj", "down_proj" but removed to lower gpu memory use
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# (using FP16 for compatibility/safety w/ legacy hardware, o/w would do BP16)
print(f"Loading base model weights in FP16 format")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

# OPTIMIZATION.. enable input gradient tracking
# must enable b4 call get_peft_mdoel when use fp16 checkpointing
model.enable_input_require_grads()
model = get_peft_model(model, lora_config)

# optimized to reduce token block bounds to lower activation sizes.. changes made marked by #opt
# training arguments and execution
training_args = TrainingArguments(
    output_dir=STAGE1_OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,
    num_train_epochs=1,  # A single epoch is sufficient to capture general summarization syntax
    logging_steps=5,
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    # force deep cache clearing on checkpoint blocks:
    gradient_checkpointing_kwargs={"use_reentrant": False},
    save_strategy="epoch",
    report_to="none",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    callbacks=[MemoryGarbageCollectorCallback()]
)

if __name__ == "__main__":
    print("Launching Stage 1 Supervised Memorization Training..")
    trainer.train()

    print(f"Saving LoRA adapter to: {STAGE1_OUTPUT_DIR}")
    model.save_pretrained(STAGE1_OUTPUT_DIR)

    # ------ IMMEDIATE merging weights block --------- #
    print("\nTraining finished! Starting weight merge process.")

    # clear out current training variables from gpu memory completely
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading base model onto CPU memory for merging:")
    base_model_merge = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda:0",  # If do cpu: Keep GPU clean during disk-heavy save phase. BUT with CPU RAM spikes, erge on L4 w/ cuda to not crash
        low_cpu_mem_usage=True,
        # needed for QWEN2.5 in FP16 (bc model natively trained in bf16..) to avoid attribute mismatch/NaN in merge
        attn_implementation="sdpa"
    )

    print("loading LoRA adapter layers:")
    peft_model = PeftModel.from_pretrained(base_model_merge, STAGE1_OUTPUT_DIR)

    print("Merging and unloading structural layers...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving permanent standalone model to: {MERGED_OUTPUT_DIR}")
    merged_model.save_pretrained(MERGED_OUTPUT_DIR)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

    # clean up merge memory before vLLM starts
    del merged_model
    del base_model_merge
    del peft_model
    gc.collect()
    torch.cuda.empty_cache()

    print("\nSUCCESS: Model fully fine-tuned, merged + GPU completely empty, ready for vLLM at 90% capacity.")


