FROM python:3.11-slim

WORKDIR /app

# uv のインストール
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 依存関係ファイルをコピー
COPY pyproject.toml uv.lock ./

# 依存関係のインストール
RUN uv sync --frozen --no-dev

# ソースコードをコピー
COPY . .

CMD ["uv", "run", "python", "main.py"]
