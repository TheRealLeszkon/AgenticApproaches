"""CLI entrypoint for the VLM dimension-estimation agent.

Examples:
    python main.py --model qwen3-vl-8b --images ./images/01.jpg ./images/02.jpg ./images/03.jpg
    python main.py --model internvl3-8b --backend mlx --images ./images/*
    python main.py --model qwen3-vl-30b --images ./images/* --output result.json

The tiny smoke-test model runs on any hardware:
    python main.py --model smolvlm-256m --images dataset/object_001/*.jpeg
"""

from __future__ import annotations

import argparse
import json
import sys

from models.base import available_models, load_model
from pipeline import run_pipeline
from utils import logging as log
from utils.config import load_config
from utils.image import expand_image_args, load_images


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate object W/H/D from 3-5 photos using an open-weight VLM.")
    p.add_argument("--model", help=f"model name; one of: {', '.join(available_models())}")
    p.add_argument("--images", nargs="+", required=True, help="image paths, globs, or a directory")
    p.add_argument("--backend", choices=["transformers", "mlx"], help="inference backend")
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    p.add_argument("--output", help="also write the final JSON to this path")
    p.add_argument("--max-image-side", type=int, help="downscale longest side above this many px")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    overrides = {"model": args.model, "backend": args.backend}
    if args.max_image_side is not None:
        overrides["max_image_side"] = args.max_image_side
    cfg = load_config(args.config, cli_overrides=overrides)

    image_paths = expand_image_args(args.images)
    if not image_paths:
        log.error("no valid JPEG/PNG images found from --images")
        return 2
    if not (3 <= len(image_paths) <= 5):
        log.warn(
            f"{len(image_paths)} image(s) supplied; this system is designed for 3-5 views. "
            "Proceeding anyway."
        )

    log.stage("Setup")
    log.info(f"Model:   {cfg.model}  (backend={cfg.backend})")
    log.info(f"Images:  {len(image_paths)} -> {', '.join(image_paths)}")
    log.info("Loading model (first run downloads weights; this can take a while)...")
    model = load_model(cfg.model, cfg)

    images = load_images(image_paths, max_image_side=cfg.max_image_side)

    final = run_pipeline(
        model, images, image_paths=image_paths, output_dir=cfg.output_dir
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(final, f, indent=2)
        log.info(f"Final JSON written to: {args.output}")

    # non-zero exit if the judge stage failed to produce valid output
    return 0 if final.get("_meta", {}).get("stage_ok", {}).get("judge", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
