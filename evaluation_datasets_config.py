import re
from llm_functions import get_model_response

######### TENGU ##########

tengu_example_question_answer = {
    "Question": "「急がば回れ」という言葉について説明してください。",
    "Answer": "「急がば回れ」という言葉は、日本の諺の一つであり、直接的な意味は「急ぐときは、早道や危険な方法を選ばずに、むしろ回り道で確実で安全な道を通った方が結局は早く着けるものだ」というものです。この言葉は、物事は慌てずに着実に進めることが結果としてうまくいくという教訓を含んでいます",
    "Criteria": "- 本来の「急ぐときは、早道や危険な方法を選ばずに、むしろ回り道で確実で安全な道を通った方が結局は早く着ける」という意味について説明している:3点\n- 一般化した「物事は慌てずに着実に進めることが結果としてうまくいく」という意味について説明している:3点\n- ことわざであることを示している:2点\n- 説明は具体的でわかりやすい:1点\n- 自然な日本語である:1点",
    "ModelAnswer": "「急がば回れ」とは、物事を急いで進めるよりも、慎重に計画を立てて行動する方が結果が良くなるという意味のことわざです。つまり、無駄なミスやトラブルを避けるためには、急いで手を打つのではなく、ゆっくりと計画を練り、周囲をよく考えて行動することが大切だということを教えています。急いで物事を進めようとして失敗してしまうよりも、手間と時間をかけてじっくりと準備をする方が結果的に効率的で成功する可能性が高いという教訓を持つ言葉です。"
}

tengu_example_evaluation = """[該当する評価項目とその簡潔な理由]
- 一般化した「物事は慌てずに着実に進めることが結果としてうまくいく」という意味について説明している:3点
  - 「物事を急いで進めるよりも、慎重に計画を立てて行動する方が結果が良くなるという意味」と示している。
- ことわざであることを示している:2点
  - 「ことわざです」と書いてある。
- 説明は具体的でわかりやすい:1点
  - 言い換えたりしながら詳しく説明されている。
- 自然な日本語である:1点
  - 日本語の用法が全て正しい。
[計算式]
3+2+1+1=7
[点数]
<score>7</score>"""

def get_tengu_prompt(data: dict) -> str:
    question = data['Question']
    example_answer = data['Answer']
    criteria = data['Criteria']
    model_answer = data['ModelAnswer']

    answer_bool = example_answer is not None
    
    example_answer = f"\n[正解例]\n{example_answer}\n" if answer_bool else ""
    answer_explanation = "'正解例'," if answer_bool else ""
    
    prompt = f"""[指示]
あなたは熟練した生成AIモデルの性能評価者です。評価項目に準拠して客観的に評価することで報酬を得ることができますが、評価項目と異なる評価をした場合報酬がもらえなくなってしまいます。
以下の'質問',{answer_explanation}'評価項目'に基づいて'評価するモデルの回答'を0~10点の整数値で評価してください。

[質問]
{question}
{example_answer}
[評価項目]
{criteria}

[評価するモデルの回答]
{model_answer}

# 以下の形式で回答してください。必ず最終的なスコアを<score>タグで囲んでください。
[該当する評価項目とその簡潔な理由]

[計算式]

[点数]
<score>点数をここに記入</score>
"""
    return prompt

def get_tengu_eval_score(eval_text: str) -> int | None:
    if eval_text is None:
        print("Received None eval_text, returning None score.")
        return None
    try:
        # Try to find score in XML tags first
        score_match = re.search(r"<score>([0-9.]+)</score>", eval_text)
        if score_match:
            return round(float(score_match.group(1)))

        # Fall back to original parsing method
        score_text_match = re.search(r"\[点数\]\\n[0-9.]+点", eval_text)
        if score_text_match:
            score_text = score_text_match.group()
            score_match_fallback = re.search(r"[0-9.]+", score_text)
            if score_match_fallback:
                score = score_match_fallback.group()
                return round(float(score))

        raise ValueError("Could not find score pattern")

    except (ValueError, AttributeError):
        print(f"Unable to parse Tengu score from '{eval_text[:100]}...'") 
        return None

def make_tengu_conversation(data: dict) -> list:
    return [
        {
            "role": "user",
            "content": get_tengu_prompt(tengu_example_question_answer)
        },{
            "role": "assistant",
            "content": tengu_example_evaluation
        },{
            "role": "user",
            "content": get_tengu_prompt(data)
        }
    ]

def tengu_bench_evaluator(data:dict, model_name:str) -> int|None:
    messages = make_tengu_conversation(data)
    evaluation = get_model_response(messages, model_name, judge=True)
    return get_tengu_eval_score(evaluation)

######### ELYZA ##########

def get_elyza_prompt(row: dict):
    question = row['Question']
    answer = row['output']
    criteria = row['eval_aspect']
    model_answer = row['ModelAnswer']
    return f"""あなたは採点者です。

問題, 正解例, 採点基準, 回答 が与えられます。

採点基準と正解例を参考にして、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。
必ず最終的なスコアを<score>タグで囲んで出力してください。

# 問題
{question}

# 正解例
{answer}

# 採点基準
基本的な採点基準
- 1点: 誤っている、 指示に従えていない
- 2点: 誤っているが、方向性は合っている
- 3点: 部分的に誤っている、 部分的に合っている
- 4点: 合っている
- 5点: 役に立つ

基本的な減点項目
- 不自然な日本語: -1点
- 部分的に事実と異なる内容を述べている: -1点
- 「倫理的に答えられません」のように過度に安全性を気にしてしまっている: 2点にする

問題固有の採点基準
{criteria}

# 回答
{model_answer}

最終スコア: <score>ここにスコアを記入</score>
"""

def elyza_evaluator(data: dict, model_name:str) -> int|None:
    prompt = get_elyza_prompt(data)
    messages = [{"role": "user", "content": prompt}]
    evaluation = get_model_response(messages, model_name, judge=True)
    try:
        # Try to find score in XML tags first
        score_match = re.search(r"<score>([0-9.]+)</score>", evaluation)
        if score_match:
            return round(float(score_match.group(1)))
            
        # Fall back to direct integer parsing
        return round(float(evaluation.strip()))
    except (ValueError, AttributeError):
        print(f"Int parse error.\n\nOutput was {evaluation}.\n\nInput was {data}.")
        return None

######### MT-Bench ##########

def get_mt_prompt(row: dict):
    question = row['Question']
    model_answer = row['ModelAnswer']
    return f"""[指示]
公平な判断者として行動し、以下に表示されるユーザーの質問に対するAIアシスタントの応答の品質を評価してください。あなたの評価は、応答の有用性、関連性、正確性、深さ、創造性、詳細度などの要素を考慮すべきです。AIアシスタントの返答の言語は、ユーザーが使用している言語と一致しているべきで、そうでない場合は減点されるべきです。評価は短い説明から始めてください。できるだけ客観的であること。説明を提供した後、このフォーマットに厳密に従って1から10までのスケールで応答を評価する必要があります："[[評価]]"、例えば："評価：[[5]]"。

[質問]
{question}

[アシスタントの回答の開始]
{model_answer}

[アシスタントの回答の終了]"""

def mt_evaluator(data: dict, model_name:str) -> int|None:
    prompt = get_mt_prompt(data)
    messages = [{"role": "user", "content": prompt}]
    evaluation = get_model_response(messages, model_name, judge=True)
    try:
        score_text = re.search(r"評価：\[\[[0-9.]+\]\]", evaluation).group()
        score = re.search(r"[0-9.]+", score_text).group()
        return round(float(score))
    except (ValueError, AttributeError):
        print(f"Int parse error.\n\nOutput was {evaluation}.\n\nInput was {data}.")
        gpt4score = None
    return gpt4score

######### Rakuda Benchmark ##########

def get_rakuda_prompt(row: dict):
    question = row['Question']
    model_answer = row['ModelAnswer']
    return f"""[指示]
以下に表示されたユーザーの質問に対するAIアシスタントのパフォーマンスについて、あなたのフィードバックをお願いします。回答の有用性、関連性、正確性、詳細度、日本語能力を評価してください。まず、アシスタントの有用性、関連性、正確性、詳細度、日本語能力の評価を提供してください。評価の包括的な説明も提供してください。ユーザーは日本語しか話さないので日本語で書かれていない回答には低評価をつけてください。できるだけ客観的であること。説明を提供した後、このフォーマットに厳密に従って1から10までのスケールで応答を評価する必要があります："[[評価]]"、例えば："評価：[[5]]"。

[質問]
{question}

[アシスタントの回答の開始]
{model_answer}

[アシスタントの回答の終了]"""

def rakuda_evaluator(data: dict, model_name:str) -> int|None:
    prompt = get_rakuda_prompt(data)
    messages = [{"role": "user", "content": prompt}]
    evaluation = get_model_response(messages, model_name, judge=True)
    try:
        score_text = re.search(r"評価：(\[\[|\[|【)[0-9.]+(\]\]|\]|】)", evaluation).group()
        score = re.search(r"[0-9.]+", score_text).group()
        return round(float(score))
    except (ValueError, AttributeError):
        print(f"Int parse error.\n\nOutput was {evaluation}.\n\nInput was {data}.")
        gpt4score = None
    return gpt4score

######### VN Translation Benchmark ##########

def get_vntl_prompt(data: dict) -> str:
    original_japanese = data.get('Question', '')
    reference_translation = data.get('reference_answer', '')
    model_translation = data.get('ModelAnswer', '')

    return f"""You are an expert in Japanese-to-English translation, specifically for visual novel content. Your task is to evaluate a model's translation based on its accuracy and naturalness compared to a reference translation.

Please score the "Model's Translation" on a scale of 1 to 10, where 10 is a perfect, natural-sounding, and accurate translation.

- **10**: Perfect. Indistinguishable from the reference; captures all nuance and sounds completely natural.
- **8-9**: Excellent. Minor differences from the reference, but still accurate and natural.
- **6-7**: Good. The meaning is correct, but the phrasing is slightly awkward or unnatural.
- **4-5**: Fair. The core meaning is mostly understandable, but contains significant grammatical errors or unnatural phrases.
- **1-3**: Poor. Largely inaccurate, nonsensical, or fails to translate the text.

Consider the context of the original Japanese dialogue.

[Original Japanese]
{original_japanese}

[Reference English Translation]
{reference_translation}

[Model's Translation to Evaluate]
{model_translation}

First, provide a brief reasoning for your score, explaining why the model's translation is good or bad. Then, provide the final score inside <score> tags.

[Reasoning]
Your reasoning here.

[Score]
<score>Your score here (integer from 1-10)</score>
"""

def vntl_evaluator(data: dict, model_name: str) -> int | None:
    prompt = get_vntl_prompt(data)
    messages = [{"role": "user", "content": prompt}]
    evaluation = get_model_response(messages, model_name, judge=True)
    try:
        score_match = re.search(r"<score>(\d+)</score>", evaluation)
        if score_match:
            return int(score_match.group(1))
        return None
    except (ValueError, AttributeError):
        print(f"Could not parse VNTL score from: {evaluation}")
        return None


EVAL_MODEL_CONFIGS = {
    "lightblue/tengu_bench": {
        "question_column": "Question",
        "evaluator_function": tengu_bench_evaluator,
        "split_name": "test"
    },
    "elyza/ELYZA-tasks-100": {
        "question_column": "input",
        "evaluator_function": elyza_evaluator,
        "split_name": "test"
    },
    "shisa-ai/ja-mt-bench-1shot": {
        "question_column": "Question",
        "evaluator_function": mt_evaluator,
        "split_name": "train"
    },
    "yuzuai/rakuda-questions": {
        "question_column": "text",
        "evaluator_function": rakuda_evaluator,
        "split_name": "train"
    },
    "lmg-anon/VNTL-v3.1-1k": {
        "question_column": "text",
        "evaluator_function": vntl_evaluator,
        "split_name": "train",
        "prompt_template": (
            "You are an expert translator specializing in Japanese visual novels. "
            "Use the following character metadata for context. Translate the entire Japanese script to natural-sounding English, maintaining the format and flow of the dialogue.\n\n"
            "### Character Metadata:\n{metadata}\n\n"
            "### Japanese Script:\n{question}\n\n"
            "### English Translation:\n"
        )
    },
}


import os

get_ans_path = lambda dataset_name, model_name: os.path.join(".", "data", "model_answers", dataset_name.replace("/", "__"), model_name.replace("/", "__") + ".json")
