# Shaberi: A Suite of Japanese Chat Benchmarks
A repo for evaluating Japanese LLMs　・　日本語LLMを評価するレポ

## What's New in This Fork
- **New Benchmark: VNTL**
  - Added a new Japanese-to-English translation benchmark using visual novel text (VNTL-Translation).
- **Environment Configuration via env.json**
  - All environment variables (such as API keys and endpoints) are now loaded from `env.json` instead of using `os.getenv`.
  - This solves issues with environment variables not propagating correctly during multiprocessing.
- **Separate OpenAI API Endpoints**
  - You can now specify separate OpenAI API-compatible endpoints for both the target model and the judge model in `env.json`.

## How to Run
```sh
# Get code
git clone https://github.com/Sub0x/shaberi
cd shaberi

# Create Environment, Install requirements
conda create -n shaberi python=3.11
conda activate shaberi
pip install -r requirements.txt

# For AMD or other hardware you may need to install torch manually
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Configure your environment
cp example.env.json env.json
# Edit env.json to add your API keys and endpoints

# In one terminal, run vLLM OpenAI API, eg:
python -m vllm.entrypoints.openai.api_server --model shisa-ai/shisa-v1-llama3-70b -tp 8
# or llama.cpp OpenAI API, eg:
./server -ngl 99 -c 8192 -m shisa-v1-llama3-70b.Q4_K_M.gguf --chat-template llama3 --host 0.0.0.0 --port 8000 -a shisa-v1-llama3-70b.q4_k_m

# In a separate terminal, generate answers:
conda activate shaberi
python generate_answers.py --model_name 'shisa-ai/shisa-v1-llama3-8b' -fp 0.5

# Then run the judge (uses endpoints and keys from env.json):
python judge_answers.py -m shisa-ai/shisa-v1-llama3-8b

# Or use the Makefile for easier command-line usage:
# Example: generate answers with options
make generate m=MODEL_NAME d=DATASET n=NUM_PROC fp=FREQ_PENALTY me=MAX_ENTRIES
# Example: judge answers with options (including evaluator)
make judge m=MODEL_NAME d=DATASET e=EVALUATOR n=NUM_PROC
# Example: filter dataset for SFW content
make filter d=DATASET b=BATCH_SIZE

# Show help for all Makefile targets
make help

# Make sure you have new answers and judgements
git status

```

## About env.json
- Place your OpenAI-compatible API keys and endpoints in `env.json`. This allows for requests to be distributed across multiple tokens if one and the following reaches rate limit.
- You can specify separate endpoints for the target model and the judge model.
- This approach ensures all processes (including those spawned via multiprocessing) have access to the correct environment variables.

Example `env.json`:
```json
{
  "OPENAI_API_KEY": [
    "sk-...",
    "sk-..."
  ],
  "JUDGE_API_KEY": [
    "sk-...",
    "sk-..."
  ],
  
  "OPENAI_API_BASE": "https://generativelanguage.googleapis.com/v1beta/openai",
  "JUDGE_API_BASE": "http://localhost:8001/v1"
}
```

## Supported Benchmarks
- ELYZA-tasks-100
- Rakuda
- Tengu-Bench
- MT-Bench
- **VNTL-Translation** (new)

## Features
- Evaluate Japanese LLMs on multiple benchmarks, including the new VNTL translation benchmark.
- Use separate endpoints for answer generation and judging.
- Robust multiprocessing support for large-scale evaluation.

## Results

<style type="text/css">
#T_aa01a_row0_col0, #T_aa01a_row0_col1, #T_aa01a_row0_col2, #T_aa01a_row0_col4, #T_aa01a_row0_col5, #T_aa01a_row2_col3 {
  background-color: #FFF8C4;
}
</style>
<table id="T_aa01a">
  <caption>Model Mean Scores by Benchmark</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_aa01a_level0_col0" class="col_heading level0 col0" >ELYZA-Tasks-100</th>
      <th id="T_aa01a_level0_col1" class="col_heading level0 col1" >Tengu-Bench-120</th>
      <th id="T_aa01a_level0_col2" class="col_heading level0 col2" >VNTL-Translation-200</th>
      <th id="T_aa01a_level0_col3" class="col_heading level0 col3" >MT-Bench-60</th>
      <th id="T_aa01a_level0_col4" class="col_heading level0 col4" >Rakuda-40</th>
      <th id="T_aa01a_level0_col5" class="col_heading level0 col5" >mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_aa01a_level0_row0" class="row_heading level0 row0" >gpt-4o</th>
      <td id="T_aa01a_row0_col0" class="data row0 col0" >9.22</td>
      <td id="T_aa01a_row0_col1" class="data row0 col1" >8.52</td>
      <td id="T_aa01a_row0_col2" class="data row0 col2" >8.28</td>
      <td id="T_aa01a_row0_col3" class="data row0 col3" >9.32</td>
      <td id="T_aa01a_row0_col4" class="data row0 col4" >9.90</td>
      <td id="T_aa01a_row0_col5" class="data row0 col5" >9.05</td>
    </tr>
    <tr>
      <th id="T_aa01a_level0_row1" class="row_heading level0 row1" >mistral-small-3.2-24b-instruct-2506</th>
      <td id="T_aa01a_row1_col0" class="data row1 col0" >8.30</td>
      <td id="T_aa01a_row1_col1" class="data row1 col1" >7.87</td>
      <td id="T_aa01a_row1_col2" class="data row1 col2" >7.35</td>
      <td id="T_aa01a_row1_col3" class="data row1 col3" >8.87</td>
      <td id="T_aa01a_row1_col4" class="data row1 col4" >9.45</td>
      <td id="T_aa01a_row1_col5" class="data row1 col5" >8.37</td>
    </tr>
    <tr>
      <th id="T_aa01a_level0_row2" class="row_heading level0 row2" >qwen3-32b</th>
      <td id="T_aa01a_row2_col0" class="data row2 col0" >8.92</td>
      <td id="T_aa01a_row2_col1" class="data row2 col1" >7.95</td>
      <td id="T_aa01a_row2_col2" class="data row2 col2" >6.83</td>
      <td id="T_aa01a_row2_col3" class="data row2 col3" >9.33</td>
      <td id="T_aa01a_row2_col4" class="data row2 col4" >8.75</td>
      <td id="T_aa01a_row2_col5" class="data row2 col5" >8.36</td>
    </tr>
    <tr>
      <th id="T_aa01a_level0_row3" class="row_heading level0 row3" >shisa-v2-qwen2.5-32b</th>
      <td id="T_aa01a_row3_col0" class="data row3 col0" >8.70</td>
      <td id="T_aa01a_row3_col1" class="data row3 col1" >7.57</td>
      <td id="T_aa01a_row3_col2" class="data row3 col2" >7.52</td>
      <td id="T_aa01a_row3_col3" class="data row3 col3" >8.78</td>
      <td id="T_aa01a_row3_col4" class="data row3 col4" >9.12</td>
      <td id="T_aa01a_row3_col5" class="data row3 col5" >8.34</td>
    </tr>
    <tr>
      <th id="T_aa01a_level0_row4" class="row_heading level0 row4" >gemma-3-27b-it</th>
      <td id="T_aa01a_row4_col0" class="data row4 col0" >8.36</td>
      <td id="T_aa01a_row4_col1" class="data row4 col1" >7.69</td>
      <td id="T_aa01a_row4_col2" class="data row4 col2" >7.46</td>
      <td id="T_aa01a_row4_col3" class="data row4 col3" >8.92</td>
      <td id="T_aa01a_row4_col4" class="data row4 col4" >9.22</td>
      <td id="T_aa01a_row4_col5" class="data row4 col5" >8.33</td>
    </tr>
    <tr>
      <th id="T_aa01a_level0_row5" class="row_heading level0 row5" >gemini-2.0-flash-lite</th>
      <td id="T_aa01a_row5_col0" class="data row5 col0" >8.30</td>
      <td id="T_aa01a_row5_col1" class="data row5 col1" >7.67</td>
      <td id="T_aa01a_row5_col2" class="data row5 col2" >7.38</td>
      <td id="T_aa01a_row5_col3" class="data row5 col3" >8.58</td>
      <td id="T_aa01a_row5_col4" class="data row5 col4" >9.60</td>
      <td id="T_aa01a_row5_col5" class="data row5 col5" >8.31</td>
    </tr>
    <tr>
      <th id="T_aa01a_level0_row6" class="row_heading level0 row6" >mistral-small-3.1-24b-instruct-2503</th>
      <td id="T_aa01a_row6_col0" class="data row6 col0" >8.30</td>
      <td id="T_aa01a_row6_col1" class="data row6 col1" >7.58</td>
      <td id="T_aa01a_row6_col2" class="data row6 col2" >7.19</td>
      <td id="T_aa01a_row6_col3" class="data row6 col3" >8.45</td>
      <td id="T_aa01a_row6_col4" class="data row6 col4" >9.18</td>
      <td id="T_aa01a_row6_col5" class="data row6 col5" >8.14</td>
    </tr>
    <tr>
      <th id="T_aa01a_level0_row7" class="row_heading level0 row7" >glm-4-32b-0414</th>
      <td id="T_aa01a_row7_col0" class="data row7 col0" >8.14</td>
      <td id="T_aa01a_row7_col1" class="data row7 col1" >7.70</td>
      <td id="T_aa01a_row7_col2" class="data row7 col2" >7.00</td>
      <td id="T_aa01a_row7_col3" class="data row7 col3" >8.53</td>
      <td id="T_aa01a_row7_col4" class="data row7 col4" >9.03</td>
      <td id="T_aa01a_row7_col5" class="data row7 col5" >8.08</td>
    </tr>
    <tr>
      <th id="T_aa01a_level0_row8" class="row_heading level0 row8" >shisa-v2-mistral-small-24b</th>
      <td id="T_aa01a_row8_col0" class="data row8 col0" >8.30</td>
      <td id="T_aa01a_row8_col1" class="data row8 col1" >7.68</td>
      <td id="T_aa01a_row8_col2" class="data row8 col2" >7.97</td>
      <td id="T_aa01a_row8_col3" class="data row8 col3" >7.93</td>
      <td id="T_aa01a_row8_col4" class="data row8 col4" >8.45</td>
      <td id="T_aa01a_row8_col5" class="data row8 col5" >8.07</td>
    </tr>
    <tr>
      <th id="T_aa01a_level0_row9" class="row_heading level0 row9" >aya-expanse-32b-abliterated</th>
      <td id="T_aa01a_row9_col0" class="data row9 col0" >8.12</td>
      <td id="T_aa01a_row9_col1" class="data row9 col1" >7.14</td>
      <td id="T_aa01a_row9_col2" class="data row9 col2" >7.49</td>
      <td id="T_aa01a_row9_col3" class="data row9 col3" >8.70</td>
      <td id="T_aa01a_row9_col4" class="data row9 col4" >8.88</td>
      <td id="T_aa01a_row9_col5" class="data row9 col5" >8.06</td>
    </tr>
    <tr>
      <th id="T_aa01a_level0_row10" class="row_heading level0 row10" >amoral-gemma3-27b-v2-qat</th>
      <td id="T_aa01a_row10_col0" class="data row10 col0" >6.46</td>
      <td id="T_aa01a_row10_col1" class="data row10 col1" >5.34</td>
      <td id="T_aa01a_row10_col2" class="data row10 col2" >6.89</td>
      <td id="T_aa01a_row10_col3" class="data row10 col3" >3.88</td>
      <td id="T_aa01a_row10_col4" class="data row10 col4" >3.52</td>
      <td id="T_aa01a_row10_col5" class="data row10 col5" >5.22</td>
    </tr>
  </tbody>
</table>

---

# 日本語版 README

## このフォークの主な変更点
- **新ベンチマーク: VNTL**
  - ビジュアルノベルのテキストを用いた日英翻訳ベンチマーク（VNTL-Translation）を追加しました。
- **env.jsonによる環境変数管理**
  - APIキーやエンドポイントなどの環境変数は `os.getenv` ではなく `env.json` から読み込むようになりました。
  - これにより、マルチプロセス時に環境変数が正しく伝播しない問題が解決されます。
- **OpenAI API互換エンドポイントの分離指定**
  - 生成モデル用・ジャッジモデル用で別々のOpenAI API互換エンドポイントを `env.json` で指定できます。

## 実行方法
```sh
# コード取得
git clone https://github.com/shisa-ai/shaberi
cd shaberi

# 環境構築・依存パッケージインストール
conda create -n shaberi python=3.11
conda activate shaberi
pip install -r requirements.txt

# AMD等の環境ではtorchを手動インストールしてください
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# 環境変数の設定
cp example.env.json env.json
# env.jsonを編集しAPIキーやエンドポイントを記入

# 別ターミナルでvLLM OpenAI APIを起動（例）
python -m vllm.entrypoints.openai.api_server --model shisa-ai/shisa-v1-llama3-70b -tp 8
# またはllama.cpp OpenAI API（例）
./server -ngl 99 -c 8192 -m shisa-v1-llama3-70b.Q4_K_M.gguf --chat-template llama3 --host 0.0.0.0 --port 8000 -a shisa-v1-llama3-70b.q4_k_m

# 別ターミナルで回答生成
conda activate shaberi
python generate_answers.py --model_name 'shisa-ai/shisa-v1-llama3-8b' -fp 0.5

# ジャッジ実行（env.jsonのエンドポイント・キーを利用）
python judge_answers.py -m shisa-ai/shisa-v1-llama3-8b

# Makefileを使ったコマンド例
make generate m=MODEL名 d=データセット名 n=NUM_PROC fp=FREQ_PENALTY me=MAX_ENTRIES
make judge m=MODEL名 d=データセット名 e=評価モデル名 n=NUM_PROC
make filter d=データセット名 b=BATCH_SIZE

# 利用可能なMakefileターゲットのヘルプ表示
make help

# 回答・評価結果の確認
git status

# 結果の集計
python results_vizualization.py
cat output.csv
```

## env.jsonについて
- OpenAI互換APIのキーやエンドポイントは `env.json` に記載してください。
- 生成用・ジャッジ用で別々のエンドポイント指定が可能です。
- マルチプロセス時も全プロセスで正しく参照されます。

`env.json`例：
```json
{
  "OPENAI_API_KEY": "sk-...",
  "OPENAI_API_BASE": "http://localhost:8000/v1",
  "JUDGE_API_KEY": "sk-...",
  "JUDGE_API_BASE": "http://localhost:8001/v1"
}
```

## 対応ベンチマーク
- ELYZA-tasks-100
- Rakuda
- Tengu-Bench
- MT-Bench
- **VNTL-Translation**（新規）

## 主な特徴
- VNTL翻訳ベンチマークを含む複数の日本語LLMベンチマークに対応
- 生成・評価で別々のAPIエンドポイント指定が可能
- 大規模評価向けの堅牢なマルチプロセス対応

---