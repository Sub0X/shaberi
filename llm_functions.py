# llm_functions.py
import backoff
import litellm
import os
import json
import random
import time
import threading

# --- Load environment variables from env.json ---
with open('env.json', 'r', encoding='utf-8') as f:
    ENV = json.load(f)
# --- end env setup ---

from datasets import Dataset
from litellm import completion

# litellm._turn_on_debug()

# Global
fp = 0.0
_key_counter = 0
_key_counter_lock = threading.Lock()

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
@backoff.on_exception(
    backoff.expo,
    (litellm.RateLimitError, litellm.InternalServerError, litellm.APIConnectionError, litellm.Timeout),
    max_tries=5,
    max_time=300
)
def get_answer_with_backoff(*args, **kwargs):
    return _get_answer_inner(*args, **kwargs)

def get_answer(question, model_name: str, judge: bool = False, prompt_template: str = None):
    return get_answer_with_backoff(question, model_name, judge, prompt_template)

def _get_answer_inner(question, model_name: str, judge: bool = False, prompt_template: str = None):
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
    generation_max_tokens = 6000

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

    # --- Round-robin key selection with retry on rate limit ---
    global _key_counter
    num_keys = len(api_keys)
    for attempt in range(num_keys):
        with _key_counter_lock:
            key_index = _key_counter % num_keys
            _key_counter += 1
        key = api_keys[key_index]
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
            response = completion(**completion_args)
            content = response.choices[0].message.content
            return content
        except litellm.RateLimitError as e:
            print(f"[WARN] Rate limit hit for key ending in '...{key[-4:]}'. Trying next key.")
            continue
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred with key ending in '...{key[-4:]}': {e}")
            # Re-raise as a standard error to avoid multiprocessing pickle issues
            raise RuntimeError(f"LLM error: {e}")
    # If all keys hit rate limit, raise to trigger backoff
    raise RuntimeError("All API keys hit rate limit.")


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