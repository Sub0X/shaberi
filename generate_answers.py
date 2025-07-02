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
    # Extract metadata
    metadata = ""
    metadata_match = re.search(r"<<METADATA>>\n(.*?)\n<<START>>", text, re.DOTALL)
    if metadata_match:
        metadata = metadata_match.group(1).strip()

    # Extract all dialogue pairs
    pairs = re.findall(r"<<JAPANESE>>\n(.*?)\n<<ENGLISH>>\n(.*?)\n", text, re.DOTALL)
    
    if not pairs:
        return None

    # Concatenate all dialogues into single blocks
    japanese_script = "\n".join([ja.strip() for ja, en in pairs])
    english_script = "\n".join([en.strip().replace('</s>', '').strip() for ja, en in pairs])
    
    return {
        "metadata": metadata, 
        "JAPANESE_SCRIPT": japanese_script, 
        "ENGLISH_SCRIPT": english_script
    }


def run_generate(model_name: str, eval_dataset_name: str = "all", num_proc: int = 16):
    MAX_ENTRIES = 400

    eval_dataset_names = list(EVAL_MODEL_CONFIGS.keys()) if eval_dataset_name == "all" else [eval_dataset_name]

    for dataset_name in eval_dataset_names:
        print(f"Loading dataset: {dataset_name}")
        eval_config = EVAL_MODEL_CONFIGS[dataset_name]
        split_name = eval_config.get("split_name", "test")
        dataset = load_dataset(dataset_name, split=split_name)
        
        # Limit to the first 200 entries if desired (applies to the original dataset rows)
        if len(dataset) > MAX_ENTRIES:
            dataset = dataset.select(range(MAX_ENTRIES))
            print(f"Limiting to the first {len(dataset)} source entries for processing.")

        # Special handling for VNTL to process as full scripts
        if dataset_name == "lmg-anon/VNTL-v3.1-1k":
            processed_entries = [process_vntl_entry_as_script(row['text']) for row in dataset]
            # Filter out any entries that might have failed parsing
            processed_entries = [entry for entry in processed_entries if entry is not None]
            
            processed_dataset = Dataset.from_list(processed_entries)

            prompt_template = eval_config.get("prompt_template")
            answer_function = get_answerer(model_name, judge=False, prompt_template=prompt_template)

            print(f"From {dataset_name}, generating answers for {model_name}...")
            # The lambda now passes the full script to the answer function
            final_dataset = processed_dataset.map(
                lambda x: {"ModelAnswer": answer_function(
                    question={"metadata": x["metadata"], "question": x["JAPANESE_SCRIPT"]},
                    model_name=model_name
                )},
                num_proc=num_proc
            )
            # Rename columns to what the judge script expects
            final_dataset = final_dataset.rename_column("JAPANESE_SCRIPT", "Question")
            final_dataset = final_dataset.rename_column("ENGLISH_SCRIPT", "reference_answer")

        # Standard handling for all other datasets
        else:
            q_col = eval_config.get("question_column")
            if q_col != "Question":
                dataset = dataset.rename_column(q_col, "Question")

            print(f"From {dataset_name}, generating answers for {model_name}...")
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
    parser.add_argument('-fp', '--frequency_penalty', type=float, default=1.0)
    args = parser.parse_args()

    run_generate(args.model_name, args.eval_dataset_name, args.num_proc)

if __name__ == '__main__':
    main()