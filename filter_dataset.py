# filter_dataset_ml.py
import json
import os
import argparse
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from tqdm import tqdm

def load_keywords_from_file(filepath):
    """Loads a list of keywords from a text file, one per line."""
    if not os.path.exists(filepath):
        print(f"Warning: Keyword file not found at {filepath}. Skipping keyword filter.")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        keywords = [line.strip() for line in f.readlines() if line.strip()]
    return keywords

def create_sfw_dataset_with_ml(dataset_name: str, batch_size: int):
    """
    Loads a dataset and applies a three-stage filtering process:
    1. A fast keyword-based filter on the original Japanese text.
    2. An English-language NSFW model on the reference text.
    3. A multilingual toxicity model on the Japanese dialogue.
    """
    # --- Define Models ---
    english_nsfw_model = "TostAI/nsfw-text-detection-large"
    toxicity_model = "textdetox/glot500-toxicity-classifier"

    # --- Define Output Path ---
    output_dir = os.path.join(".", "datasets")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = dataset_name.replace("/", "__") + ".jsonl"
    output_filepath = os.path.join(output_dir, output_filename)
    
    # --- Load Models ---
    print("Loading ML models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load English NSFW classifier using pipeline
    english_classifier = pipeline("text-classification", model=english_nsfw_model, device=0 if device=="cuda" else -1)
    
    # Load Toxicity classifier manually
    toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model)
    toxicity_model_instance = AutoModelForSequenceClassification.from_pretrained(toxicity_model)
    toxicity_model_instance.to(device)
    toxicity_model_instance.eval()
    
    print("Loading original dataset from Hugging Face...")
    dataset = load_dataset(dataset_name, split='train')

    # --- Stage 1: Keyword Pre-filtering ---
    keyword_filepath = os.path.join("assets", "profanity_list.txt")
    print(f"\n--- Stage 1: Keyword Filtering ---")
    nsfw_keywords = load_keywords_from_file(keyword_filepath)
    stage1_filtered = []
    keyword_nsfw_count = 0
    if nsfw_keywords:
        for row in tqdm(dataset, desc="Keyword Filtering"):
            if not any(keyword in row['text'] for keyword in nsfw_keywords):
                stage1_filtered.append(row)
            else:
                keyword_nsfw_count += 1
    else:
        stage1_filtered = list(dataset)
    print(f"Keyword filtering complete. {keyword_nsfw_count} entries removed.")

    if not stage1_filtered:
        print("No entries left after keyword filtering. Exiting.")
        return
        
    # --- Stage 2: English NSFW Model Filtering ---
    print(f"\n--- Stage 2: English NSFW Model Filtering ---")
    english_texts = [" ".join([en.strip().replace('</s>', '') for _, en in re.findall(r"<<JAPANESE>>\n(.*?)\n<<ENGLISH>>\n(.*?)\n", row['text'], re.DOTALL)]) for row in stage1_filtered]
    
    with tqdm(total=len(english_texts), desc="English NSFW Classification") as pbar:
        predictions_stage2 = []
        for i in range(0, len(english_texts), batch_size):
            batch = english_texts[i:i + batch_size]
            batch_preds = english_classifier(batch, truncation=True)
            predictions_stage2.extend(batch_preds)
            pbar.update(len(batch))
    
    SFW_LABELS = ['LABEL_0', 'safe']
    
    stage2_filtered = []
    english_nsfw_count = 0
    for row, pred in zip(stage1_filtered, predictions_stage2):
        if pred['label'] in SFW_LABELS:
            stage2_filtered.append(row)
        else:
            english_nsfw_count += 1
    print(f"English NSFW filtering complete. {english_nsfw_count} entries removed.")

    if not stage2_filtered:
        print("No entries left after English NSFW filtering. Exiting.")
        return

    # --- Stage 3: Japanese Toxicity Model Filtering ---
    print(f"\n--- Stage 3: Japanese Toxicity Filtering ---")
    final_sfw_entries = []
    toxicity_nsfw_count = 0
    for row in tqdm(stage2_filtered, desc="Toxicity Filtering (line-by-line)"):
        dialogue_lines = [ja.strip() for ja, _ in re.findall(r"<<JAPANESE>>\n(.*?)\n<<ENGLISH>>\n(.*?)\n", row['text'], re.DOTALL) if ja.strip()]
        if not dialogue_lines:
            final_sfw_entries.append(row)
            continue
            
        is_entry_toxic = False
        for i in range(0, len(dialogue_lines), batch_size):
            batch_texts = dialogue_lines[i:i + batch_size]
            inputs = toxicity_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = toxicity_model_instance(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1)
            
            if any(pred.item() == 1 for pred in predictions): # 1 is 'toxic'
                is_entry_toxic = True
                break
        
        if not is_entry_toxic:
            final_sfw_entries.append(row)
        else:
            toxicity_nsfw_count += 1
            
    print(f"Toxicity filtering complete. {toxicity_nsfw_count} entries removed.")

    # --- Final Summary ---
    print(f"\n--- Filtering Summary ---")
    print(f"  Original entries: {len(dataset)}")
    print(f"  Removed by keyword filter: {keyword_nsfw_count}")
    print(f"  Removed by English NSFW model: {english_nsfw_count}")
    print(f"  Removed by Japanese toxicity model: {toxicity_nsfw_count}")
    print(f"  Final SFW entries remaining: {len(final_sfw_entries)}")
    
    print(f"\nSaving SFW dataset to '{output_filepath}'...")
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for entry in final_sfw_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"✅ SFW dataset created successfully at '{output_filepath}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a dataset for SFW content using a three-stage process.")
    
    parser.add_argument('-d', '--dataset_name', type=str, default="lmg-anon/VNTL-v3.1-1k")
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    
    args = parser.parse_args()
    
    create_sfw_dataset_with_ml(args.dataset_name, args.batch_size)