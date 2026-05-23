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
docker run --rm make-llm
```

## 動作確認
```bash
uv run python main.py
```
