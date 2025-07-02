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
make generate fp=0.5 n=1 m=MODEL_NAME d=DATASET
# Example: judge answers with options (including evaluator)
make judge fp=0.5 n=1 m=MODEL_NAME d=DATASET e=EVALUATOR

# Make sure you have new answers and judgements
git status

```

## About env.json
- Place your OpenAI-compatible API keys and endpoints in `env.json`.
- You can specify separate endpoints for the target model and the judge model.
- This approach ensures all processes (including those spawned via multiprocessing) have access to the correct environment variables.

Example `env.json`:
```json
{
  "OPENAI_API_KEY": "sk-...",
  "OPENAI_API_BASE": "http://localhost:8000/v1",
  "JUDGE_API_KEY": "sk-...",
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
make generate fp=0.5 n=1 m=MODEL名 d=データセット名
make judge fp=0.5 n=1 m=MODEL名 d=データセット名 e=評価モデル名

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