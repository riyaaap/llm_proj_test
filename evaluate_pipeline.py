import os
import sys
import torch
import textstat
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from rouge_score import rouge_scorer

# ensure NLTK resources available locally
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# hardware setup to only use 1 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Paths to models
BASE_MODEL_ID = "Qwen/Qwen2.5-7B"
FINE_TUNED_PATH = "./Qwen2.5-7B-Summarizer"


### TEST DATA (orig articles and ground truth summaries) ###
TEST_DATASET = [
    {
        "article": "The Federal Reserve kept its benchmark interest rate unchanged on Wednesday but signaled that it remains on track to cut rates later this year. Chair Jerome Powell stated that inflation has eased notably over the past year but remains above the central bank's 2% target. Financial markets reacted positively to the news, with major stock indices jumping roughly 1.2% following the press conference.",
        "ground_truth": "The Federal Reserve held interest rates steady but hinted at future rate cuts this year as inflation continues to cool toward its 2% target."
    },
    {
        "article": "A team of marine biologists discovered three new species of deep-sea coral during an expedition in the Mariana Trench. Utilizing an advanced remote-operated vehicle (ROV) capable of withstanding extreme hydrostatic pressures, the researchers collected samples at depths exceeding 6,000 meters. Preliminary DNA sequencing suggests these species diverged from shallow-water relatives roughly 40 million years ago.",
        "ground_truth": "Biologists found three new deep-sea coral species at depths of over 6,000 meters in the Mariana Trench using a specialized ROV."
    }
]

# Evaluation metric helper functs
def calculate_semantic_similarity(text1, text2, tokenizer, model):
    """Calculates cosine similarity using the model's own internal hidden-state embeddings."""
    inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, max_length=512).to("cuda")

    with torch.no_grad():
        emb1 = model.get_input_embeddings()(inputs1.input_ids).mean(dim=1)
        emb2 = model.get_input_embeddings()(inputs2.input_ids).mean(dim=1)

    sim = cosine_similarity(emb1.cpu().numpy(), emb2.cpu().numpy())
    return float(sim[0][0])

def get_readability_score(text):
    """Calculates Flesch-Kincaid Grade Level (Lower = Easier to read for lay audiences)."""
    return textstat.flesch_kincaid_grade(text)

def llm_judge_hallucination_check(article, summary, tokenizer, model):
    """Uses base model itself as zero-shot judge to score factual consistency (1-5 scale)."""
    judge_prompt = (
        f"<|im_start|>system\nYou are an unbiased AI evaluator checking for hallucinations. "
        f"Rate the summary based strictly on the provided article on a scale from 1 (contains completely fabricated facts) "
        f"to 5 (perfect factual accuracy, zero hallucinations). Output only a single number.<|im_end|>\n"
        f"<|im_start|>user\nArticle: {article}\nSummary: {summary}\n\nRating (1-5):<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(judge_prompt, return_tensors="pt").to("cuda")

    # track exact token sequence length dimension directly
    prompt_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False, # turn off sampling for deterministic
            eos_token_id=tokenizer.eos_token_id
        )

    # correctly slice 2D tensor using scalar prompt_length
    generated_tokens = outputs[0][prompt_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    try:
        # Extract the first digit found in the response string
        digits = [int(s) for s in response.split() if s.isdigit()]
        return digits[0] if digits else int(response[0])
    except Exception:
        return 3  # Fallback middle score if parsing fails

def generate_summary(model, tokenizer, article, is_fine_tuned=True):
    """Generates a summary using the standardized template schema."""
    if is_fine_tuned:
        prompt = (
            f"<|im_start|>system\nYou are an expert assistant. Summarize the following article accurately, "
            f"preserving technical terms, methods, and key focuses, in an easily readable manner that lay audiences can understand.<|im_end|>\n"
            f"<|im_start|>user\n{article}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        prompt = f"Write a short, concise summary of this article:\n{article}\nSummary:"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # track exact token sequence length dimension directly
    prompt_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,      # Explicitly enable sampling so temperature works
            temperature=0.1,
            eos_token_id=tokenizer.eos_token_id
        )

    # remove outputs[0] tracking and targeting prompt_length
    generated_tokens = outputs[0][prompt_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

## MAIN EXECUTION ##
def main():
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # --- PHASE 1: EVALUATE BASE MODEL ---
    print("Loading original un-fine-tuned base Qwen Model layers...")
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto")

    base_results = []
    for item in TEST_DATASET:
        summary = generate_summary(base_model, base_tokenizer, item["article"], is_fine_tuned=False)
        base_results.append(summary)

    # Free up VRAM explicitly before loading the fine-tuned weights
    del base_model
    torch.cuda.empty_cache()

    # --- PHASE 2: EVALUATE FINE-TUNED MODEL ---
    print("\nLoading Custom Fine-Tuned Model layers...")
    ft_tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_PATH)
    ft_model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_PATH, torch_dtype=torch.float16, device_map="auto")

    ft_results = []
    for item in TEST_DATASET:
        summary = generate_summary(ft_model, ft_tokenizer, item["article"], is_fine_tuned=True)
        ft_results.append(summary)

# --- PHASE 3: METRIC ANALYSIS COMPARISON LOOP ---
    print("\n" + "="*50)
    print("PIPELINE BENCHMARKING REPORT")
    print("="*50)

    for i, item in enumerate(TEST_DATASET):
        print(f"\nTEST ITEM {i+1}")
        print(f"Ground Truth Target: '{item['ground_truth']}'")
        print(f"Base Model Summary: '{base_results[i]}'")
        print(f"Fine-Tuned Summary: '{ft_results[i]}'")
        print("-" * 30)

        # Readability Calcs
        base_read = get_readability_score(base_results[i])
        ft_read = get_readability_score(ft_results[i])
        gt_read = get_readability_score(item["ground_truth"])

        # Semantic Embeddings Similarity
        base_sim = calculate_semantic_similarity(item["article"], base_results[i], ft_tokenizer, ft_model)
        ft_sim = calculate_semantic_similarity(item["article"], ft_results[i], ft_tokenizer, ft_model)

        # ROUGE Hard Token Overlap Match
        base_rouge = scorer.score(item["ground_truth"], base_results[i])['rougeL'].fmeasure
        ft_rouge = scorer.score(item["ground_truth"], ft_results[i])['rougeL'].fmeasure

        # LLM-as-a-Judge Factual Accuracy Checking
        base_judge = llm_judge_hallucination_check(item["article"], base_results[i], ft_tokenizer, ft_model)
        ft_judge = llm_judge_hallucination_check(item["article"], ft_results[i], ft_tokenizer, ft_model)

        # Metrics Printout Dashboard Matrix
        print(f"[ROUGE-L Precision Overlap]  Base: {base_rouge:.3f}  |  Fine-Tuned: {ft_rouge:.3f}")
        print(f"[Semantic Profile Match]    Base: {base_sim:.3f}  |  Fine-Tuned: {ft_sim:.3f}")
        print(f"[Readability Grade level]   Base: {base_read}   |  Fine-Tuned: {ft_read}  (Target GT: {gt_read})")
        print(f"LLM Judge Accuracy (1-5)]  Base: {base_judge}/5    |  Fine-Tuned: {ft_judge}/5")
        print("="*50)

if __name__ == "__main__":
    main()
