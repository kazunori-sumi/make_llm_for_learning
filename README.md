# 作りながら学ぶ!LLM入門

## 開発環境

| 項目 | バージョン |
|---|---|
| Python | 3.11 |
| PyTorch | 2.x |
| NumPy | 2.4.x |
| tiktoken | 0.12.x |
| パッケージ管理 | [uv](https://docs.astral.sh/uv/) |

## セットアップ（ローカル）

```bash
# uv のインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync

# 実行
uv run python main.py
```

## セットアップ（Docker）

```bash
# イメージのビルド
docker build -t make-llm .

# 実行
docker run --rm -v $(pwd):/app -v /app/.venv make-llm
```

Apple Silicon などの ARM64 環境で実行時に何も表示されない場合は、PyTorch の CPU 命令互換性の問題を避けるため amd64 イメージとしてビルド・実行してください。

```bash
docker build --platform linux/amd64 -t make-llm-amd64 .
docker run --rm --platform linux/amd64 -v $(pwd):/app -v /app/.venv make-llm-amd64
```

## 動作確認
```bash
uv run python main.py
```
