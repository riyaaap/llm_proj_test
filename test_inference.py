import os
import sys

# Force execution strictly on GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Target local merged weights directory
MODEL_PATH = "./Qwen2.5-7B-Summarizer"

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: The directory '{MODEL_PATH}' does not exist.")
        print("Please check the merged output folder name.")
        sys.exit(1)

    print("Loading tokenizer and model weights directly onto GPU 1 (FP16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Load model in native FP16, matches fine-tuning pipeline
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model weights successfully allocated in VRAM!")

    # Provide a sample text snippet to test the summarization syntax
    sample_article = (
        "The National Aeronautics and Space Administration (NASA) has successfully "
        "launched its latest climate monitoring satellite into orbit. The spacecraft, "
        "equipped with advanced hyperspectral imaging sensors, will track ocean surface "
        "temperatures and greenhouse gas concentrations globally. Researchers state that "
        "the raw data collected will be processed using high-performance cloud clusters "
        "and made entirely open-source to the scientific community within three months."
    )

    # Wrap input in exact training prompt structure
    prompt = (
        f"<|im_start|>system\nYou are an expert assistant. "
        f"Summarize the following article accurately, preserving technical terms, "
        f"methods, and key focuses, in an easily readable manner that lay audiences can understand.<|im_end|>\n"
        f"<|im_start|>user\n{sample_article.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    print("\nRunning text generation pass...")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate tokens using greedy decoding (temperature=0 to isolate behavior)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    # Slice the output tokens to isolate just the generated summary response
    generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    summary_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("\n" + "="*40)
    print("CUSTOM FINE-TUNED MODEL SUMMARY OUTPUT:")
    print("="*40)
    print(summary_output.strip())
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
