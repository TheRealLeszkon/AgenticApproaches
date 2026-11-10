"""Qwen3-VL adapter (Transformers backend).

Covers qwen3-vl-8b and qwen3-vl-30b-a3b. Uses the standard chat-template flow. We keep
the generic ``AutoModelForImageTextToText`` loader so this works across transformers
versions; if your version exposes a dedicated class you may set MODEL_CLASS to it.

For local Apple Silicon, prefer the MLX backend (see models/mlx_backend.py):
    --model qwen3-vl-8b --backend mlx
"""

from __future__ import annotations

from models._transformers_chat import ChatTemplateModel


class Qwen3VLModel(ChatTemplateModel):
    MODEL_CLASS = "AutoModelForImageTextToText"
    TRUST_REMOTE_CODE = False
