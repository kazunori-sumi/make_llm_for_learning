# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLMの仕組みを学ぶためのPythonプロジェクト。PyTorchを使ってTransformer/Attentionの基礎から実装する学習用リポジトリ。

学習目的のため、基本的には実装を依頼されない限り、例やヒントの提示にとどめてください

学習者はpythonの基礎がないため、丁寧な説明をお願いいたします

回答は日本語でお願いいたします

## Commands

```bash
# 依存関係のインストール
uv sync

# 実行
uv run python main.py
# または仮想環境を有効化して
source .venv/bin/activate
python main.py
```

## Architecture

- **Package manager:** UV (`uv.lock` でロック)
- **Python:** 3.11+ (`.python-version` で指定)
- **Key dependencies:** PyTorch 2.10, NumPy 2.4

現在 `main.py` のみで構成。クエリベクトルと入力ベクトル群のdot product計算（Attentionの基礎）を実装している。
