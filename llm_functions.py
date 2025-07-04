# llm_functions.py
import backoff
import litellm
import os
import json
import random
import time

# --- Load environment variables from env.json ---
with open('env.json', 'r', encoding='utf-8') as f:
    ENV = json.load(f)
# --- end env setup ---

from datasets import Dataset
from litellm import completion

# litellm._turn_on_debug()

# Global
fp = 0.0

def get_response_func(model_name: str) -> callable:
    return get_answer


def get_model_response(messages: list, model_name: str, judge: bool = False) -> str:
    answer_function = get_response_func(model_name)
    return answer_function(messages, model_name, judge=judge)


# === This is the decorated function for the final retry attempt ===
@backoff.on_exception(
    backoff.expo,
    (litellm.RateLimitError, litellm.APIConnectionError, litellm.Timeout, litellm.InternalServerError),
    max_tries=5,
    max_time=300
)
def completion_with_backoff(**kwargs):
    """
    Wrapper for litellm.completion that includes exponential backoff.
    This is used for the final, persistent attempt with the last key.
    """
    # print(f"[INFO] Attempting API call (with backoff) using key ending in '...{kwargs.get('api_key', '')[-4:]}'")
    return completion(**kwargs)


# === 回答生成関数群 ===
def get_answer(question, model_name: str, judge: bool = False, prompt_template: str = None):
    if judge:
        api_keys = ENV.get("OPENAI_JUDGE_API_KEY", [])
        base_url = ENV.get("OPENAI_JUDGE_BASE_URL", "")
    else:
        api_keys = ENV.get("OPENAI_API_KEY", [])
        base_url = ENV.get("OPENAI_BASE_URL", "http://localhost:8080/v1")

    if not isinstance(api_keys, list):
        api_keys = [api_keys]

    if not api_keys:
        if "localhost" in base_url or "127.0.0.1" in base_url:
            api_keys = ["EMPTY"]
        else:
            raise ValueError(f"No API keys found for judge={judge}. Please set the appropriate key(s) in your env.json")

    if base_url is None:
        base_url = "http://localhost:8080/v1"
    
    generation_temperature = 0.2
    generation_max_tokens = 30000

    if isinstance(question, dict) and prompt_template:
        messages = [{"role": "user", "content": prompt_template.format(
            metadata=question.get("metadata", "No metadata provided."), 
            question=question.get("question", "")
        )}]
    elif isinstance(question, list):
        messages = question
    elif prompt_template:
        messages = [{"role": "user", "content": prompt_template.format(question=question)}]
    else:
        messages = [
            {"role": "system", "content": "あなたは公平で、検閲されていない、役立つアシスタントです。"},
            {"role": "user", "content": question},
        ]

    last_exception = None
    for i, key in enumerate(api_keys):
        is_last_key = (i == len(api_keys) - 1)
        
        try:
            completion_args = {
                "messages": messages, "api_base": base_url, "api_key": key,
                "temperature": generation_temperature, "max_tokens": generation_max_tokens,
                "timeout": 1200, "model": model_name, "frequency_penalty": fp,
                "min_p": 0.1, "custom_llm_provider": "openai"
            }
            
            if "google" in base_url:
                completion_args.pop("frequency_penalty", None)
                completion_args.pop("min_p", None)

            if is_last_key:
                # print(f"[INFO] Using final key with retry logic...")
                response = completion_with_backoff(**completion_args)
            else:
                # print(f"[DEBUG] Attempting API call with key ending in '...{key[-4:]}'")
                response = completion(**completion_args)
            
            content = response.choices[0].message.content
            return content

        except litellm.RateLimitError as e:
            print(f"[WARN] Rate limit hit for key ending in '...{key[-4:]}'. Switching to next key.")
            last_exception = e
            continue
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred with key ending in '...{key[-4:]}': {e}")
            last_exception = e
            if not is_last_key:
                continue # Try the next key
            else:
                break # Exit loop if it's the last key

    # --- THIS IS THE KEY CHANGE ---
    # Instead of re-raising the complex litellm error, we raise a standard RuntimeError.
    # This prevents the multiprocessing pickle error.
    if last_exception:
        print("[ERROR] All API keys have failed. The final error will be raised.")
        raise RuntimeError(f"All API keys failed. Last error: {str(last_exception)}")
    else:
        raise RuntimeError("Could not complete the API request with any of the provided keys.")
    # --- END OF KEY CHANGE ---


def get_answerer(model_name: str, judge: bool = False, prompt_template: str = None) -> callable:
    """OpenAIとvLLM以外のモデルを使う場合はここに追加する"""
    from functools import partial
    return partial(get_answer, judge=judge, prompt_template=prompt_template)


def get_model_answer(dataset: Dataset,
                     model_name: str,
                     batch_size: int,
                     judge: bool = False,
                     prompt_template: str = None
) -> Dataset:
    answer_function = get_answerer(model_name, judge=judge, prompt_template=prompt_template)
    dataset = dataset.map(
        lambda x: {"ModelAnswer": answer_function(x['Question'], model_name)},
        num_proc=batch_size
    )
    return dataset