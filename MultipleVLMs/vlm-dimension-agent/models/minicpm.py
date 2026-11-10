"""MiniCPM-V 4.5 adapter (Transformers backend).

MiniCPM-V uses remote code and a bespoke ``model.chat(msgs=..., tokenizer=...)`` API where
``content`` is a flat list mixing PIL images and text. This adapter wires all input images
followed by the text prompt into a single user turn.

Requires: transformers, torch, torchvision, sentencepiece  (pip install ...[remote])
Needs trust_remote_code=True. On Apple Silicon prefer the MLX backend.

NOTE: not exercised by the local smoke test; follows the official model card API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from models.base import VisionModel
from models._common import resolve_device, resolve_dtype, quantization_config

if TYPE_CHECKING:
    from PIL.Image import Image
    from utils.config import Config


class MiniCPMModel(VisionModel):
    def __init__(self, repo: str, cfg: "Config"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.repo = repo
        self.cfg = cfg
        self.device = resolve_device(cfg)
        self.dtype = resolve_dtype(cfg, self.device)
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        load_kwargs = {
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
        }
        qcfg = quantization_config(cfg)
        if qcfg is not None:
            load_kwargs["quantization_config"] = qcfg
            load_kwargs["device_map"] = "auto"
        elif self.device == "cuda":
            load_kwargs["device_map"] = "auto"

        self.model = AutoModel.from_pretrained(repo, **load_kwargs)
        if "device_map" not in load_kwargs:
            self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, images: list["Image"], prompt: str) -> str:
        content = list(images) + [prompt]
        msgs = [{"role": "user", "content": content}]

        kwargs = {}
        if self.cfg.temperature and self.cfg.temperature > 0:
            kwargs.update(sampling=True, temperature=self.cfg.temperature)
        else:
            kwargs.update(sampling=False)

        with self._torch.no_grad():
            response = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                max_new_tokens=self.cfg.max_tokens,
                **kwargs,
            )
        return response.strip() if isinstance(response, str) else str(response)
