# llm_functions.py
import backoff
import openai
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
from openai import OpenAI

# Global
fp = 0.0
_key_counter = 0
_key_counter_lock = threading.Lock()

def get_response_func(model_name: str) -> callable:
    return get_answer


def get_model_response(messages: list, model_name: str, judge: bool = False, temperature: float = None, top_k: int = None, top_p: float = None, repetition_penalty: float = None) -> str:
    answer_function = get_response_func(model_name)
    return answer_function(messages, model_name, judge=judge, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)


# === This is the decorated function for the final retry attempt ===
@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError),
    max_tries=5,
    max_time=300
)
def completion_with_backoff(**kwargs):
    """
    Wrapper for openai client completion that includes exponential backoff.
    This is used for the final, persistent attempt with the last key.
    """
    # print(f"[INFO] Attempting API call (with backoff) using key ending in '...{kwargs.get('api_key', '')[-4:]}'")
    client = OpenAI(api_key=kwargs.pop('api_key'), base_url=kwargs.pop('api_base'))
    return client.chat.completions.create(**kwargs)


# === 回答生成関数群 ===
@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError, openai.APITimeoutError),
    max_tries=5,
    max_time=300
)
def get_answer_with_backoff(*args, **kwargs):
    return _get_answer_inner(*args, **kwargs)

def get_model_response(question, model_name: str, judge: bool = False, prompt_template: str = None, temperature: float = None, top_p: float = None):
    return get_answer_with_backoff(question, model_name, judge, prompt_template, temperature=temperature, top_p=top_p)

def _get_answer_inner(question, model_name: str, judge: bool = False, prompt_template: str = None, temperature: float = None, top_p: float = None):
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
    
    # Use provided temperature if given, otherwise fall back to the existing default
    generation_temperature = 0.2 if temperature is None else float(temperature)
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

    # --- Round-robin key selection with retry on rate limit, repeating rounds until success or backoff ---
    global _key_counter
    num_keys = len(api_keys)
    while True:
        last_exception = None
        for attempt in range(num_keys):
            with _key_counter_lock:
                key_index = _key_counter % num_keys
                _key_counter += 1
            key = api_keys[key_index]
            try:
                # Create OpenAI client for this request
                client = OpenAI(api_key=key, base_url=base_url, timeout=1200)
                
                completion_args = {
                    "messages": messages,
                    "temperature": generation_temperature,
                    "max_tokens": generation_max_tokens,
                    "model": model_name,
                }
                
                # Add optional parameters if provided
                if top_p is not None:
                    completion_args["top_p"] = top_p
                
                # Only add these parameters if not using Google
                if "google" not in base_url:
                    completion_args["frequency_penalty"] = fp
                    # Note: min_p is not a standard OpenAI parameter, removing it
                
                try:
                    response = client.chat.completions.create(**completion_args)
                    content = response.choices[0].message.content
                    return content
                except TypeError as e:
                    if "unexpected keyword argument" in str(e):
                        print(f"[WARN] Parameter not supported by API: {str(e).split('argument ')[-1].strip()}")
                        print(f"[DEBUG] Attempted parameters: {[k for k in completion_args.keys() if k != 'messages']}")
                        # Remove the unsupported parameter and retry
                        unsupported_param = str(e).split('argument ')[-1].strip("'")
                        retry_args = {k: v for k, v in completion_args.items() if k != unsupported_param}
                        print(f"[DEBUG] Retrying without '{unsupported_param}': {[k for k in retry_args.keys() if k != 'messages']}")
                        response = client.chat.completions.create(**retry_args)
                        content = response.choices[0].message.content
                        return content
                    else:
                        raise
            except openai.UnprocessableEntityError as e:
                print(f"[WARN] Content policy violation (422) - skipping this evaluation")
                print(f"[WARN] Model: {model_name}, Message count: {len(messages)}")
                raise RuntimeError(f"Content policy violation (422): {e}")
            except openai.RateLimitError as e:
                print(f"[WARN] Rate limit hit for key ending in '...{key[-4:]}'. Trying next key.")
                last_exception = e
                continue
            except Exception as e:
                if isinstance(e, (openai.InternalServerError, openai.APIConnectionError, openai.APITimeoutError)):
                    last_exception = e
                    raise  # Let backoff handle these
                
                # Log the specific request that failed for debugging
                if hasattr(e, 'status_code') and e.status_code == 422:
                    print(f"[DEBUG] 422 Error Details:")
                    print(f"[DEBUG] Model: {model_name}")
                    print(f"[DEBUG] Messages: {json.dumps(messages, indent=2, ensure_ascii=False)}")
                    print(f"[DEBUG] Temperature: {generation_temperature}")
                    print(f"[DEBUG] Max tokens: {generation_max_tokens}")
                
                print(f"[ERROR] An unexpected error occurred with key ending in '...{key[-4:]}': {e}")
                raise RuntimeError(f"LLM error: {e}")
        # If all keys hit rate limit, repeat the round (backoff will still limit total retries)
        if last_exception:
            print("[INFO] All API keys hit rate limit in this round. Retrying all keys...")
            continue
        raise RuntimeError("All API keys hit rate limit.")


def get_answerer(model_name: str, judge: bool = False, prompt_template: str = None, temperature: float = None, top_p: float = None) -> callable:
    """OpenAIとvLLM以外のモデルを使う場合はここに追加する"""
    from functools import partial
    return partial(get_model_response, judge=judge, prompt_template=prompt_template, temperature=temperature, top_p=top_p)


def get_model_answer(dataset: Dataset,
                     model_name: str,
                     batch_size: int,
                     judge: bool = False,
                     prompt_template: str = None,
                     temperature: float = None,
                     top_p: float = None
) -> Dataset:
    answer_function = get_answerer(model_name, judge=judge, prompt_template=prompt_template, temperature=temperature, top_p=top_p)
    dataset = dataset.map(
        lambda x: {"ModelAnswer": answer_function(x['Question'], model_name)},
        num_proc=batch_size
    )
    return dataset