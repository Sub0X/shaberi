import re
from llm_functions import get_model_response
import orjson

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
あなたは学術的な言語モデル評価の専門家です。研究目的で、モデルの回答品質を客観的に評価してください。評価項目に準拠して客観的に評価することで報酬を得ることができますが、評価項目と異なる評価をした場合報酬がもらえなくなってしまいます。
以下の'質問',{answer_explanation}'評価項目'に基づいて'評価するモデルの回答'を0~10点の整数値で評価してください。

注意：この評価は学術研究目的であり、内容の適切性ではなく、回答の品質（正確性、完全性、明確性）のみを評価してください。

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

def tengu_bench_evaluator(data:dict, model_name:str, temperature: float = None, retry_attempt: bool = False) -> int|None:
    messages = make_tengu_conversation(data)
    
    # On retry, add emphatic format instructions
    if retry_attempt:
        messages.append({
            "role": "user", 
            "content": "**CRITICAL: You MUST follow the exact format and include the <score> tags with a numerical score. This is your final attempt.**"
        })
    
    evaluation = get_model_response(messages, model_name, judge=True, temperature=temperature)
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

def elyza_evaluator(data: dict, model_name:str, temperature: float = None, retry_attempt: bool = False) -> int|None:
    prompt = get_elyza_prompt(data)
    
    # On retry, add emphatic format instructions
    if retry_attempt:
        prompt += "\n\n**CRITICAL: You MUST provide a score in <score> tags. This is your final attempt.**"
    
    messages = [{"role": "user", "content": prompt}]
    evaluation = get_model_response(messages, model_name, judge=True, temperature=temperature)
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

def mt_evaluator(data: dict, model_name:str, temperature: float = None, retry_attempt: bool = False) -> int|None:
    prompt = get_mt_prompt(data)
    
    # On retry, add emphatic format instructions
    if retry_attempt:
        prompt += "\n\n**CRITICAL: You MUST provide your evaluation in the exact format '評価：[[数字]]'. This is your final attempt.**"
    
    messages = [{"role": "user", "content": prompt}]
    evaluation = get_model_response(messages, model_name, judge=True, temperature=temperature)
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

def rakuda_evaluator(data: dict, model_name:str, temperature: float = None, retry_attempt: bool = False) -> int|None:
    prompt = get_rakuda_prompt(data)
    
    # On retry, add emphatic format instructions
    if retry_attempt:
        prompt += "\n\n**CRITICAL: You MUST provide your evaluation in the exact format '評価：[[数字]]' or '評価：[数字]'. This is your final attempt.**"
    
    messages = [{"role": "user", "content": prompt}]
    evaluation = get_model_response(messages, model_name, judge=True, temperature=temperature)
    try:
        score_text = re.search(r"評価：(\[\[|\[|【)[0-9.]+(\]\]|\]|】)", evaluation).group()
        score = re.search(r"[0-9.]+", score_text).group()
        return round(float(score))
    except (ValueError, AttributeError):
        print(f"Int parse error.\n\nOutput was {evaluation}.\n\nInput was {data}.")
        gpt4score = None
    return gpt4score


######### VN Translation Benchmark ##########

def get_vntl_prompt_multi_score(data: dict) -> str:
    """
    Creates a detailed prompt for the LLM judge, asking it to evaluate a translation
    against a reference based on a strict protocol.
    """
    original_japanese = data.get('Question', '')
    reference_translation = data.get('reference_answer', '')
    model_translation = data.get('ModelAnswer', '')
    metadata = data.get('metadata', 'No metadata provided.')

    return f"""You are an expert evaluator for Japanese-to-English visual novel translations. Your task is to score the "Model's Translation" on six criteria by comparing it to the reference translation and the original Japanese.

### **Evaluation Criteria & Scoring (Scale 1-10):**

1.  **Accuracy (1-10):** How faithful is the translation to the original Japanese meaning?
    * **9-10 (Excellent):** Flawlessly conveys the original meaning and all subtleties.
    * **7-8 (Good):** The core meaning is correct, but minor nuances may be lost.
    * **5-6 (Fair):** The main point is translated, but there are noticeable inaccuracies.
    * **3-4 (Poor):** Significant parts of the meaning are lost or mistranslated.
    * **1-2 (Very Poor):** Completely misunderstands or fails to translate the source text.

2.  **Fluency (1-10):** How natural and grammatically correct is the English?
    * **9-10 (Excellent):** Reads like it was written by a native English speaker; grammatically perfect.
    * **7-8 (Good):** Only minor grammatical awkwardness that doesn't impede understanding.
    * **5-6 (Fair):** Contains noticeable grammatical errors or unnatural phrasing.
    * **3-4 (Poor):** Difficult to read due to grammatical mistakes.
    * **1-2 (Very Poor):** Largely ungrammatical or nonsensical.

3.  **Character Voice (1-10):** Does the translation reflect the character's personality and speaking style?
    * **9-10 (Excellent):** The character's unique personality, speech patterns, and formality level are perfectly captured.
    * **7-8 (Good):** The voice is mostly correct but may miss some subtle character-specific tics.
    * **5-6 (Fair):** The character is recognizable, but the translation often feels generic.
    * **3-4 (Poor):** The translation uses a voice that is inconsistent with or contradicts the character.
    * **1-2 (Very Poor):** The character's voice is completely lost or misrepresented.

4.  **Tone (1-10):** Does the translation capture the emotional mood of the scene?
    * **9-10 (Excellent):** The emotional mood (e.g., tension, humor, sadness) is perfectly conveyed.
    * **7-8 (Good):** The general tone is correct, but some emotional subtleties are missed.
    * **5-6 (Fair):** The tone is often flat or doesn't fully match the scene's emotional context.
    * **3-4 (Poor):** The tone is mismatched with the scene (e.g., sounds cheerful during a serious moment).
    * **1-2 (Very Poor):** The tone is completely wrong and undermines the scene's intent.

5.  **Localization (1-10):** How well are cultural nuances, idioms, and onomatopoeia handled?
    * **9-10 (Excellent):** Cultural nuances are handled cleverly, and onomatopoeia (e.g., "doki doki") is preserved correctly and naturally.
    * **7-8 (Good):** Nuances and onomatopoeia are handled correctly but might feel slightly forced.
    * **5-6 (Fair):** Attempts to handle nuances but may be literal or slightly confusing. Onomatopoeia might be incorrectly translated instead of transliterated.
    * **3-4 (Poor):** Cultural references are misunderstood or ignored.
    * **1-2 (Very Poor):** Fails completely to address cultural specifics.

6.  **Direction Following (1-10):** How strictly did the model follow the formatting protocol?
    * **9-10 (Perfect):** Flawlessly adheres to all rules. All dialogue uses the exact `SPEAKER: "DIALOGUE"` format, all honorifics are preserved, and there is **zero** added commentary or formatting.
    * **7-8 (Minor Errors):** Mostly compliant, but with very minor, occasional errors (e.g., a single instance of a missing space after the colon, or incorrect punctuation). The intent to follow the rules is clear.
    * **5-6 (Noticeable Errors):** Several formatting errors are present (e.g., inconsistent spacing, occasionally forgets quotes, mixes dialogue formats).
    * **3-4 (Major Errors):** The model regularly fails to use the required `SPEAKER: "DIALOGUE"` format, often omits honorifics, or adds some conversational filler.
    * **1-2 (No Adherence):** The model completely ignores the formatting rules, adding its own tags (like 'said' or 'replied'), inventing speakers, or failing to format dialogue correctly at all.
---
[Context & Metadata]
{metadata}

[Original Japanese Script]
{original_japanese}

[Reference English Translation (For context on quality)]
{reference_translation}

[Model's Translation to Evaluate]
{model_translation}

---
First, provide a brief reasoning for your scores based on the criteria. Then, provide all six scores as a JSON object inside <scores_json> tags.

[Reasoning]
Please provide a brief reasoning for your scores here.

[Scores]
<scores_json>
{{
  "score_accuracy": INTEGER_FROM_1_TO_10,
  "score_fluency": INTEGER_FROM_1_TO_10,
  "score_character_voice": INTEGER_FROM_1_TO_10,
  "score_tone": INTEGER_FROM_1_TO_10,
  "score_localization": INTEGER_FROM_1_TO_10,
  "score_direction_following": INTEGER_FROM_1_TO_10
}}
</scores_json>
"""

def vntl_multi_score_evaluator(data: dict, model_name: str, temperature: float = None, retry_attempt: bool = False) -> dict | None:
    """
    Calls the LLM judge with the detailed VNTL prompt and parses the six-part score.
    """
    prompt = get_vntl_prompt_multi_score(data)
    
    # On retry, add emphatic format instructions
    if retry_attempt:
        prompt += "\n\n**CRITICAL: You MUST follow the exact format. Your response MUST include ALL SIX scores in the JSON format inside <scores_json> tags. DO NOT deviate from this format or skip any scores. This is your final attempt.**"
    
    messages = [{"role": "user", "content": prompt}]
    
    evaluation = get_model_response(messages, model_name, judge=True, temperature=temperature)
    
    try:
        # Use regex to find the JSON block. re.DOTALL allows '.' to match newlines.
        json_match = re.search(r"<scores_json>(.*?)</scores_json>", evaluation, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            scores = orjson.loads(json_str)
            # Check for all six expected score keys
            expected_keys = [
                "score_accuracy", "score_fluency", "score_character_voice", 
                "score_tone", "score_localization", "score_direction_following"
            ]
            if all(key in scores for key in expected_keys):
                return scores
        raise ValueError("Could not find all required score keys in the JSON block.")
    except (orjson.JSONDecodeError, AttributeError, ValueError) as e:
        print(f"Could not parse VNTL scores from evaluation. Error: {e}")
        print(f"Evaluation content:\n{evaluation}")
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
        "evaluator_function": vntl_multi_score_evaluator,
        "split_name": "train",
        "prompt_template": (
            "You are an expert visual novel translator. Translate the following Japanese script into natural English. "
            "Follow these strict rules:\n"
            "1.  **Dialogue Format:** If a speaker is present, the output MUST be `SPEAKER_NAME: \"DIALOGUE\"`. For narration with no speaker, output only the translated text.\n"
            "2.  **No Commentary:** Do NOT add any extra text, explanations, or dialogue tags like 'said' or 'asked'.\n"
            "3.  **Preserve Elements:** Keep all Japanese honorifics (e.g., -san, -sama) and transliterate onomatopoeia (e.g., 'doki doki').\n\n"
            "### Character Metadata:\n{metadata}\n\n"
            "### Japanese Script:\n{question}\n\n"
            "### English Translation:\n"
        )
    },
}


import os

get_ans_path = lambda dataset_name, model_name: os.path.join(".", "data", "model_answers", dataset_name.replace("/", "__"), model_name.replace("/", "__") + ".json")
