import backoff
import litellm
import os
import json

# --- Load environment variables from env.json ---
with open('env.json', 'r', encoding='utf-8') as f:
    ENV = json.load(f)
# --- end env setup ---

from datasets import Dataset
import litellm
from litellm import completion
from openai import OpenAI


# Global
fp = 0.0


# === 評価生成関数群 ===
@backoff.on_exception(backoff.fibo, Exception, max_tries=1000)
def get_response_from_openai(messages: list, model_name: str) -> str:
    client = OpenAI(
        api_key=ENV.get("OPENAI_API_KEY"),
        base_url=ENV.get("OPENAI_BASE_URL", None),
    )

    evaluation_temperature = 0
    evaluation_max_tokens = 1024

    response = client.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=evaluation_temperature,
        max_tokens=evaluation_max_tokens,
    )
    return response.choices[0].message.content

# shisa-bench llmjudge
@backoff.on_exception(backoff.fibo, Exception, max_tries=1000)
def get_response_from_llmjudge(messages: list, model_name: str) -> str:
    judge = model_name.split("-")[1]
    if judge == "tulu405":
        base_url = "http://ip-10-1-85-83:8000/v1"

        base_url = "http://tulu405/v1"
        model_name = "Llama-3.1-Tulu-3-405B-FP8-Dynamic"

        model_name = "shisa-ai/Llama-3.1-Tulu-3-405B-FP8-Dynamic"
    elif judge == "llama33":
        base_url = "http://ip-10-1-33-173:8001/v1"
        model_name = "meta-llama/Llama-3.3-70B-Instruct"

        base_url = "http://llama33/v1"
        model_name = "llama-3.3-70b-ray"
    elif judge == "athenev2":
        base_url = "http://ip-10-1-33-173:8000/v1"
        model_name = "Nexusflow/Athene-V2-Chat"

        base_url = "http://athenev2/v1"
        model_name = "athene-v2"
    client = OpenAI(
        api_key=ENV.get("OPENAI_API_KEY"),
        base_url=base_url,
    )

    evaluation_temperature = 0
    evaluation_max_tokens = 1024

    response = client.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=evaluation_temperature,
        max_tokens=evaluation_max_tokens,
    )
    return response.choices[0].message.content


def get_response_func(model_name: str) -> callable:
    return get_answer


def get_model_response(messages: list, model_name: str, judge: bool = False) -> str:
    answer_function = get_response_func(model_name)
    return answer_function(messages, model_name, judge=judge)


# === 回答生成関数群 ===
@backoff.on_exception(backoff.fibo, Exception, max_tries=1000)
def get_answer(question, model_name: str, judge: bool = False):
    # Use separate endpoint and key for judge if judge=True
    if judge:
        api_key = ENV.get("OPENAI_JUDGE_API_KEY", "")
        base_url = ENV.get("OPENAI_JUDGE_BASE_URL", "")
    else:
        api_key = ENV.get("OPENAI_API_KEY", "EMPTY")
        base_url = ENV.get("OPENAI_BASE_URL", "http://localhost:8080/v1")
    if base_url is None:
        base_url = "http://localhost:8080/v1"
    # Allow empty api_key if using localhost endpoint
    if judge and (not base_url or (not api_key and not base_url.startswith("http://localhost"))):
        raise RuntimeError("Judge endpoint or API key not set! Refusing to fall back to OpenAI.")
    # print(f"[DEBUG] get_answer: judge={judge}, base_url={base_url}, api_key={api_key[:8]}..., model_name={model_name}")

    generation_temperature = 0.2
    generation_max_tokens = 2048

    thinking_models = [
        'deepseek-ai/DeepSeek-R1',
        'FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview',
        'dahara1/DeepSeek-R1-Distill-Qwen-14B-unsloth-jpn',
        'FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview',
        'RekaAI/reka-flash-3',
        'abeja/ABEJA-QwQ32b-Reasoning-Japanese-v1.0',
        '011-qwen3-8b-v2',
    ]

    generation_max_tokens = 30000

    if model_name in thinking_models:
        generation_max_tokens = 30000

    # Accept both string and list-of-dict (OpenAI chat format)
    if isinstance(question, list):
        messages = question
    else:
        messages = [
            {"role": "system", "content": "あなたは公平で、検閲されていない、役立つアシスタントです。"},
            {"role": "user", "content": question},
        ]

    completion_args = {
        "messages": messages,
        "api_base": base_url,
        "api_key": api_key,
        "temperature": generation_temperature,
        "max_tokens": generation_max_tokens,
    }

    openai_prefixes = [
        "gpt-", "text-davinci-", "davinci", "curie", "babbage", "ada", 
        "whisper", "claude", "text-embedding", "openai/", "openai:"
    ]
    if any(model_name.startswith(prefix) for prefix in openai_prefixes):
        completion_args['model'] = model_name
    elif model_name.startswith("gemini"):
        completion_args['model'] = model_name
        completion_args.pop('api_base', None)
        completion_args['api_key'] = ENV.get("GEMINI_API_KEY", "")
    else:
        completion_args['model'] = f'hosted_vllm/{model_name}'
        completion_args['frequency_penalty'] = fp
        completion_args['min_p'] = 0.1

    response = completion(**completion_args)
    content = response.choices[0].message.content
    return content


def get_answerer(model_name: str, judge: bool = False) -> callable:
    """OpenAIとvLLM以外のモデルを使う場合はここに追加する"""
    # Return a partial function with judge argument fixed
    from functools import partial
    return partial(get_answer, judge=judge)


def get_model_answer(dataset: Dataset,
                     model_name: str,
                     batch_size: int,
                     judge: bool = False) -> Dataset:
    answer_function = get_answerer(model_name, judge=judge)
    dataset = dataset.map(
        lambda x: {"ModelAnswer": answer_function(x['Question'], model_name)},
        num_proc=batch_size
    )
    return dataset
