"""InternVL3 adapter (Transformers backend).

InternVL3 does not use the HF chat-template/processor flow. It ships remote code and is
driven via ``model.chat(tokenizer, pixel_values, question, generation_config, ...)`` with
its own dynamic image-tiling preprocessing. This adapter reproduces the documented
preprocessing (see the InternVL model card) and wires multiple images with
``Image-i: <image>`` markers.

Requires: transformers, torch, torchvision, einops, timm  (pip install ...[remote])
Needs trust_remote_code=True. On Apple Silicon prefer the MLX backend.

NOTE: this path targets a CUDA/large-RAM machine and is not exercised by the local
smoke test. The preprocessing below follows the official model card exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from models.base import VisionModel
from models._common import resolve_device, resolve_dtype, quantization_config

if TYPE_CHECKING:
    from PIL.Image import Image
    from utils.config import Config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff = float("inf")
    best = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target)
        if diff < best_diff or (
            diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
        ):
            best_diff = diff
            best = ratio
    return best


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    w, h = image.size
    aspect_ratio = w / h
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, w, h, image_size)
    target_w = image_size * ratio[0]
    target_h = image_size * ratio[1]
    blocks = ratio[0] * ratio[1]
    resized = image.resize((target_w, target_h))

    tiles = []
    cols = target_w // image_size
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


class InternVL3Model(VisionModel):
    def __init__(self, repo: str, cfg: "Config"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.repo = repo
        self.cfg = cfg
        self.device = resolve_device(cfg)
        self.dtype = resolve_dtype(cfg, self.device)
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            repo, trust_remote_code=True, use_fast=False
        )
        load_kwargs = {"torch_dtype": self.dtype, "trust_remote_code": True, "low_cpu_mem_usage": True}
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

    def _pixel_values(self, image, max_num=12):
        transform = _build_transform(448)
        tiles = _dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=max_num)
        pv = [transform(t) for t in tiles]
        return self._torch.stack(pv).to(self.dtype).to(self.device)

    def generate(self, images: list["Image"], prompt: str) -> str:
        torch = self._torch
        pixel_values_list = []
        num_patches_list = []
        for img in images:
            pv = self._pixel_values(img)
            pixel_values_list.append(pv)
            num_patches_list.append(pv.shape[0])
        pixel_values = torch.cat(pixel_values_list, dim=0)

        # Prefix each image with an "Image-i: <image>" marker (multi-image convention).
        image_prefix = "".join(
            f"Image-{i + 1}: <image>\n" for i in range(len(images))
        )
        question = image_prefix + prompt

        gen_config = {"max_new_tokens": self.cfg.max_tokens, "do_sample": bool(self.cfg.temperature)}
        if self.cfg.temperature:
            gen_config["temperature"] = self.cfg.temperature

        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                gen_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            )
        return response.strip() if isinstance(response, str) else str(response)
