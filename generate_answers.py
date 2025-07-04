# generate_answers.py
import argparse
import os
import re
import json
from datasets import Dataset, load_dataset
from evaluation_datasets_config import EVAL_MODEL_CONFIGS, get_ans_path
from llm_functions import get_answerer

def process_vntl_entry_as_script(text: str):
    """
    Processes a single entry from the VNTL dataset, treating all its dialogues as one script.
    """
    metadata = ""
    metadata_match = re.search(r"<<METADATA>>\n(.*?)\n<<START>>", text, re.DOTALL)
    if metadata_match:
        metadata = metadata_match.group(1).strip()
    pairs = re.findall(r"<<JAPANESE>>\n(.*?)\n<<ENGLISH>>\n(.*?)\n", text, re.DOTALL)
    if not pairs:
        return None
    japanese_script = "\n".join([ja.strip() for ja, en in pairs])
    english_script = "\n".join([en.strip().replace('</s>', '').strip() for ja, en in pairs])
    return {
        "metadata": metadata, 
        "JAPANESE_SCRIPT": japanese_script, 
        "ENGLISH_SCRIPT": english_script
    }

def run_generate(model_name: str, eval_dataset_name: str = "all", num_proc: int = 16, max_entries: int = 200):
    """
    Generates answers for evaluation datasets using the specified model.
    Args:
        model_name (str): The name of the model to use for generating
        eval_dataset_name (str): The name of the evaluation dataset to use.
        num_proc (int): Number of processes to use for parallel processing.
    """
    MAX_ENTRIES = max_entries
    eval_dataset_names = list(EVAL_MODEL_CONFIGS.keys()) if eval_dataset_name == "all" else [eval_dataset_name]

    for dataset_name in eval_dataset_names:
        # Construct the expected output path first
        model_answer_path = get_ans_path(dataset_name, model_name)
        
        # Check if the file already exists
        if os.path.exists(model_answer_path):
            print(f"Answers for '{model_name}' on '{dataset_name}' already exist. Skipping.")
            continue # Move to the next benchmark in the loop

        
        eval_config = EVAL_MODEL_CONFIGS[dataset_name]

        # --- Check for local file before downloading ---
        local_filepath = os.path.join("datasets", dataset_name.replace("/", "__") + ".jsonl")
        if os.path.exists(local_filepath):
            print(f"Found local dataset file. Loading from: '{local_filepath}'")
            dataset = load_dataset('json', data_files=local_filepath, split='train', keep_in_memory=True)
        else:
            print(f"Local file not found. Loading '{dataset_name}' from Hugging Face Hub...")
            split_name = eval_config.get("split_name", "test")
            dataset = load_dataset(dataset_name, split=split_name, keep_in_memory=True)
        
        # Limit entries after loading, regardless of source
        if len(dataset) > MAX_ENTRIES:
            dataset = dataset.select(range(MAX_ENTRIES))
            print(f"Limiting to the first {len(dataset)} source entries for processing.")

        # Special handling for VNTL to process as full scripts
        if dataset_name == "lmg-anon/VNTL-v3.1-1k":
            print("VNTL dataset detected. Processing each entry as a full script...")
            processed_entries = [process_vntl_entry_as_script(row['text']) for row in dataset]
            processed_entries = [entry for entry in processed_entries if entry is not None]
            processed_dataset = Dataset.from_list(processed_entries)
            prompt_template = eval_config.get("prompt_template")
            answer_function = get_answerer(model_name, judge=False, prompt_template=prompt_template)
            print(f"Generating answers for {model_name}...")
            final_dataset = processed_dataset.map(
                lambda x: {"ModelAnswer": answer_function(
                    question={"metadata": x["metadata"], "question": x["JAPANESE_SCRIPT"]},
                    model_name=model_name
                )},
                num_proc=num_proc
            )
            final_dataset = final_dataset.rename_column("JAPANESE_SCRIPT", "Question")
            final_dataset = final_dataset.rename_column("ENGLISH_SCRIPT", "reference_answer")
        else:
            # Standard handling for all other datasets
            q_col = eval_config.get("question_column")
            if q_col != "Question":
                dataset = dataset.rename_column(q_col, "Question")
            print(f"Generating answers for {model_name}...")
            answer_function = get_answerer(model_name, judge=False)
            final_dataset = dataset.map(
                lambda x: {"ModelAnswer": answer_function(x['Question'], model_name)},
                num_proc=num_proc
            )
        
        # Save the final results
        model_answer_path = get_ans_path(dataset_name, model_name)
        os.makedirs(os.path.dirname(model_answer_path), exist_ok=True)
        final_dataset.to_json(model_answer_path)
        print(f"Saved answers to {model_answer_path}\\n")

def main():
    parser = argparse.ArgumentParser(description='Generate model answers for evaluation benchmarks.')
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('-d', '--eval_dataset_name', type=str, default='all')
    parser.add_argument('-n', '--num_proc', type=int, default=8)
    parser.add_argument('-fp', '--frequency_penalty', type=float, default=0.5)
    parser.add_argument('-me', '--max_entries', type=int, default=200)
    args = parser.parse_args()
    run_generate(args.model_name, args.eval_dataset_name, args.num_proc, args.max_entries)

if __name__ == '__main__':
    main()