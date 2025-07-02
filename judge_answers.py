# judge_answers.py

import argparse
import os
import re
import json
import pandas as pd
import sacrebleu
from datasets import load_dataset
from evaluation_datasets_config import EVAL_MODEL_CONFIGS, get_ans_path


reasoning_pattern = re.compile(r'<(think|thinking|reason)>.*?</(think|thinking|reason)>', re.DOTALL | re.IGNORECASE)

def evaluate_with_llm(model_name: str, eval_dataset_name: str, evaluation_model: str, num_proc: int):
    """
    Evaluates answers using a Language Model as a judge.
    """
    model_answer_path = get_ans_path(eval_dataset_name, model_name)
    # Make sure to load the reference answer from the generated file
    ans_dataset = load_dataset('json', data_files=model_answer_path, split="train")
    
    eval_config = EVAL_MODEL_CONFIGS.get(eval_dataset_name)
    eval_fn = eval_config["evaluator_function"]

    # Remove reasoning/thinking segments from answers before judging
    ans_dataset = ans_dataset.map(
        lambda x: {"ModelAnswer": reasoning_pattern.sub("", x.get("ModelAnswer", ""))},
        num_proc=num_proc,
    )
    
    ans_dataset = ans_dataset.map(lambda x: {"score": eval_fn(x, evaluation_model)}, num_proc=num_proc)

    output_dir = os.path.join(".", "data", "judgements", f"judge_{evaluation_model.replace('/', '__')}", eval_dataset_name.replace("/", "__"))
    os.makedirs(output_dir, exist_ok=True)
    ans_dataset.to_json(os.path.join(output_dir, model_name.replace("/", "__") + ".json"))
    print(f"✅ LLM-based judgements saved to {output_dir}")

def evaluate_with_metrics(model_name: str, eval_dataset_name: str):
    """
    Evaluates translation answers using BLEU and chrF metrics.
    """
    answers_file = get_ans_path(eval_dataset_name, model_name)
    
    print(f"Loading generated answers from: {answers_file}")
    with open(answers_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    model_answers = [item.get('ModelAnswer', '') for item in data]
    reference_answers = [[item.get('reference_answer', '') for item in data]]

    print("Calculating corpus-level scores...")
    bleu = sacrebleu.corpus_bleu(model_answers, reference_answers)
    chrf = sacrebleu.corpus_chrf(model_answers, reference_answers)

    print("\n--- Metric-Based Results ---")
    print(f"  Model: {model_name} on {eval_dataset_name}")
    print(f"  BLEU Score: {bleu.score:.2f}")
    print(f"  chrF Score: {chrf.score:.2f}")
    print("---------------------------------")

    for item in data:
        item['bleu'] = sacrebleu.sentence_bleu(item.get('ModelAnswer', ''), [item.get('reference_answer', '')]).score
        item['chrf'] = sacrebleu.sentence_chrf(item.get('ModelAnswer', ''), [item.get('reference_answer', '')]).score

    output_dir = os.path.join(".", "data", "judgements", "metrics", eval_dataset_name.replace("/", "__"))
    os.makedirs(output_dir, exist_ok=True)
    results_file_path = os.path.join(output_dir, model_name.replace("/", "__") + ".csv")
    
    pd.DataFrame(data).to_csv(results_file_path, index=False, encoding='utf-8-sig')
    print(f"✅ Metric-based judgements saved to {results_file_path}")


def run_judgement(model_name: str, eval_dataset_name: str = "all", evaluation_model: str = "gpt-4-turbo-preview", num_proc: int = 8):
    eval_dataset_names = list(EVAL_MODEL_CONFIGS.keys()) if eval_dataset_name == "all" else [eval_dataset_name]
    
    for eval_ds_name in eval_dataset_names:
        eval_config = EVAL_MODEL_CONFIGS.get(eval_ds_name)
        if not eval_config:
            print(f"Warning: No config found for {eval_ds_name}. Skipping.")
            continue
            
        evaluation_type = eval_config.get("evaluation_type", "llm") # Default to 'llm'

        if evaluation_type == "metric":
            print(f"Judging {model_name} on {eval_ds_name} using metrics")
            evaluate_with_metrics(model_name, eval_ds_name)
        elif "evaluator_function" in eval_config:
            print(f"Judging {model_name} on {eval_ds_name} using judge model: {evaluation_model}")
            evaluate_with_llm(model_name, eval_ds_name, evaluation_model, num_proc)
        else:
            print(f"Warning: No valid evaluation method found for {eval_ds_name}. Skipping.")

def main():
    parser = argparse.ArgumentParser(description='Judge model answers with LLM as judge or with metrics.')
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('-d', '--eval_dataset_name', type=str, default='all')
    parser.add_argument('-e', '--evaluation_model', type=str, default='gpt-4.1')
    parser.add_argument('-n', '--num_proc', type=int, default=8)
    args = parser.parse_args()
    
    run_judgement(args.model_name, args.eval_dataset_name, args.evaluation_model, args.num_proc)

if __name__ == '__main__':
    main()