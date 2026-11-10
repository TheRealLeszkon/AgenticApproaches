"""Qwen2.5-VL adapter (Transformers backend).

Covers qwen2.5-vl-7b. Standard chat-template flow. Recent transformers exposes
``Qwen2_5_VLForConditionalGeneration``; we default to the generic auto class for
portability but fall back gracefully.

For local Apple Silicon, prefer the MLX backend:
    --model qwen2.5-vl-7b --backend mlx
"""

from __future__ import annotations

from models._transformers_chat import ChatTemplateModel


class Qwen25VLModel(ChatTemplateModel):
    MODEL_CLASS = "AutoModelForImageTextToText"
    TRUST_REMOTE_CODE = False
