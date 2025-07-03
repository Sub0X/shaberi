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

# litellm._turn_on_debug()

# Global
fp = 0.0

def get_response_func(model_name: str) -> callable:
    return get_answer


def get_model_response(messages: list, model_name: str, judge: bool = False) -> str:
    answer_function = get_response_func(model_name)
    return answer_function(messages, model_name, judge=judge)


# === 回答生成関数群 ===
# @backoff.on_exception(backoff.fibo, Exception, max_tries=1000)
def get_answer(question, model_name: str, judge: bool = False, prompt_template: str = None):
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
    
    # print(f"[DEBUG] get_answer: judge={judge}, base_url={base_url}, api_key={api_key}..., model_name={model_name}")

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

    if isinstance(question, dict) and prompt_template:
        # Handle the new VNTL format with metadata
        messages = [{"role": "user", "content": prompt_template.format(
            metadata=question.get("metadata", "No metadata provided."), 
            question=question.get("question", "")
        )}]
    elif isinstance(question, list):
        # Handle existing list-based conversations
        messages = question
    elif prompt_template:
        # Handle simple prompts with a template
        messages = [{"role": "user", "content": prompt_template.format(question=question)}]
    else:
        # Default behavior for other benchmarks
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
        "timeout": 1200,
        "model": model_name,
        "frequency_penalty": fp,
        "min_p": 0.1,
        "custom_llm_provider": "openai"
    }

    response = completion(**completion_args)
    content = response.choices[0].message.content
    return content


def get_answerer(model_name: str, judge: bool = False, prompt_template: str = None) -> callable:
    """OpenAIとvLLM以外のモデルを使う場合はここに追加する"""
    # Return a partial function with judge argument fixed
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
