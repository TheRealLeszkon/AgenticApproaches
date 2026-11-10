"""SmolVLM adapter (Transformers) — tiny smoke-test model.

SmolVLM-256M is an Idefics3-style model. It uses the standard chat-template flow, so it
just inherits ChatTemplateModel. Its purpose here is to validate the whole 4-agent
pipeline end-to-end on any hardware — accuracy is not expected to be good.
"""

from __future__ import annotations

from models._transformers_chat import ChatTemplateModel


class SmolVLMModel(ChatTemplateModel):
    MODEL_CLASS = "AutoModelForImageTextToText"
    TRUST_REMOTE_CODE = False
