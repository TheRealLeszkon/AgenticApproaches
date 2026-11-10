"""Base adapter for Transformers models that use the standard chat-template flow.

Qwen2.5-VL, Qwen3-VL and SmolVLM all work the same way:

    messages -> processor.apply_chat_template(...) -> processor(text, images) -> model.generate

Families with bespoke inference APIs (InternVL3, MiniCPM-V) do NOT use this base; they
implement ``generate`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from models.base import VisionModel
from models._common import resolve_device, resolve_dtype, quantization_config

if TYPE_CHECKING:
    from PIL.Image import Image
    from utils.config import Config


class ChatTemplateModel(VisionModel):
    #: import path of the model class to load; default is the generic auto class.
    MODEL_CLASS = "AutoModelForImageTextToText"
    #: whether the family's remote code must be trusted
    TRUST_REMOTE_CODE = False

    def __init__(self, repo: str, cfg: "Config"):
        import transformers

        self.repo = repo
        self.cfg = cfg
        self.device = resolve_device(cfg)
        self.dtype = resolve_dtype(cfg, self.device)

        model_cls = getattr(transformers, self.MODEL_CLASS)
        processor_cls = transformers.AutoProcessor

        self.processor = processor_cls.from_pretrained(
            repo, trust_remote_code=self.TRUST_REMOTE_CODE
        )

        load_kwargs: dict = {
            "torch_dtype": self.dtype,
            "trust_remote_code": self.TRUST_REMOTE_CODE,
        }
        qcfg = quantization_config(cfg)
        if qcfg is not None:
            load_kwargs["quantization_config"] = qcfg
            load_kwargs["device_map"] = "auto"
        elif self.device == "cuda":
            load_kwargs["device_map"] = "auto"

        self.model = model_cls.from_pretrained(repo, **load_kwargs)
        if "device_map" not in load_kwargs:
            self.model = self.model.to(self.device)
        self.model.eval()

    def _build_messages(self, images: list["Image"], prompt: str) -> list[dict]:
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def generate(self, images: list["Image"], prompt: str) -> str:
        import torch

        messages = self._build_messages(images, prompt)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        gen_kwargs = {"max_new_tokens": self.cfg.max_tokens}
        if self.cfg.temperature and self.cfg.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.cfg.temperature)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        # decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = out[:, input_len:]
        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        return decoded[0].strip()
