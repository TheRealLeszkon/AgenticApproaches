"""Benchmark the pipeline against a labeled dataset.

Dataset layout:
    dataset/
      object_001/
        01.jpg 02.jpg 03.jpg
        ground_truth.json   ->  {"width_cm": 120, "height_cm": 180, "depth_cm": 45}
      object_002/ ...

Metrics per dimension: absolute error, percentage error, MAPE (across objects), plus an
overall MAPE, the range-hit rate (how often the true value falls inside the predicted
[min,max]), and a confidence-vs-error summary.

    python benchmark.py --model qwen3-vl-8b --dataset ./dataset
    python benchmark.py --model smolvlm-256m --dataset ./dataset --backend transformers
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from models.base import available_models, load_model
from pipeline import run_pipeline
from utils import logging as log
from utils.config import load_config
from utils.image import expand_image_args, load_images

DIMS = ["width_cm", "height_cm", "depth_cm"]


def _find_objects(dataset_dir: str) -> list[str]:
    out = []
    for name in sorted(os.listdir(dataset_dir)):
        d = os.path.join(dataset_dir, name)
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "ground_truth.json")):
            out.append(d)
    return out


def _pct_error(pred: float | None, truth: float | None) -> float | None:
    if pred is None or truth in (None, 0):
        return None
    return abs(pred - truth) / abs(truth) * 100.0


def _in_range(truth: float | None, rng) -> bool | None:
    if truth is None or not isinstance(rng, (list, tuple)) or len(rng) != 2 or None in rng:
        return None
    lo, hi = min(rng), max(rng)
    return lo <= truth <= hi


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def evaluate(dataset_dir: str, cfg) -> dict:
    objects = _find_objects(dataset_dir)
    if not objects:
        raise SystemExit(f"no objects with ground_truth.json found under {dataset_dir}")

    model = load_model(cfg.model, cfg)

    per_object = []
    for obj_dir in objects:
        with open(os.path.join(obj_dir, "ground_truth.json")) as f:
            truth = json.load(f)
        image_paths = expand_image_args([obj_dir])
        images = load_images(image_paths, max_image_side=cfg.max_image_side)

        log.stage(f"Object: {os.path.basename(obj_dir)}")
        final = run_pipeline(
            model, images, image_paths=image_paths,
            output_dir=cfg.output_dir, run_name=f"bench_{os.path.basename(obj_dir)}",
        )
        pred = final.get("dimensions", {}) or {}
        ranges = final.get("ranges", {}) or {}
        conf = final.get("confidence", {}) or {}

        row = {"object": os.path.basename(obj_dir)}
        for d in DIMS:
            p, t = pred.get(d), truth.get(d)
            row[f"{d}_pred"] = p
            row[f"{d}_true"] = t
            row[f"{d}_abs_err"] = (abs(p - t) if (p is not None and t is not None) else None)
            row[f"{d}_pct_err"] = _pct_error(p, t)
            row[f"{d}_in_range"] = _in_range(t, ranges.get(d))
            row[f"{d}_conf"] = conf.get(d.replace("_cm", ""))
        per_object.append(row)

    # aggregate
    summary = {"model": cfg.model, "n_objects": len(per_object), "per_dimension": {}}
    all_pct = []
    for d in DIMS:
        pcts = [r[f"{d}_pct_err"] for r in per_object]
        hits = [r[f"{d}_in_range"] for r in per_object if r[f"{d}_in_range"] is not None]
        summary["per_dimension"][d] = {
            "mape": _mean(pcts),
            "mae": _mean([r[f"{d}_abs_err"] for r in per_object]),
            "range_hit_rate": (sum(hits) / len(hits) if hits else None),
        }
        all_pct.extend([p for p in pcts if p is not None])
    summary["overall_mape"] = _mean(all_pct)

    # confidence vs error: mean predicted confidence vs mean pct error, per dimension
    summary["confidence_vs_error"] = {
        d: {
            "mean_confidence": _mean([r[f"{d}_conf"] for r in per_object]),
            "mean_pct_error": _mean([r[f"{d}_pct_err"] for r in per_object]),
        }
        for d in DIMS
    }
    return {"summary": summary, "per_object": per_object}


def print_report(summary: dict) -> None:
    print(f"\nModel: {summary['model']}   (objects: {summary['n_objects']})\n")
    names = {"width_cm": "Width", "height_cm": "Height", "depth_cm": "Depth"}
    for d in DIMS:
        s = summary["per_dimension"][d]
        mape = f"{s['mape']:.1f}%" if s["mape"] is not None else "n/a"
        rhr = f"{s['range_hit_rate'] * 100:.0f}%" if s["range_hit_rate"] is not None else "n/a"
        print(f"{names[d]}:\n  MAPE: {mape}\n  Range-hit: {rhr}\n")
    overall = f"{summary['overall_mape']:.1f}%" if summary["overall_mape"] is not None else "n/a"
    print(f"Overall:\n  MAPE: {overall}\n")
    print("Confidence vs error (per dimension):")
    for d in DIMS:
        c = summary["confidence_vs_error"][d]
        mc = f"{c['mean_confidence']:.2f}" if c["mean_confidence"] is not None else "n/a"
        me = f"{c['mean_pct_error']:.1f}%" if c["mean_pct_error"] is not None else "n/a"
        print(f"  {names[d]:6s} mean confidence {mc}  vs  mean error {me}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark VLM dimension estimates against ground truth.")
    p.add_argument("--model", help=f"model name; one of: {', '.join(available_models())}")
    p.add_argument("--dataset", required=True, help="dataset directory (contains object_* folders)")
    p.add_argument("--backend", choices=["transformers", "mlx"])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output", help="write full results JSON here (default: <output_dir>/benchmark_results.json)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    cfg = load_config(args.config, cli_overrides={"model": args.model, "backend": args.backend})

    results = evaluate(args.dataset, cfg)
    print_report(results["summary"])

    out_json = args.output or os.path.join(cfg.output_dir, "benchmark_results.json")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    out_csv = os.path.splitext(out_json)[0] + ".csv"
    if results["per_object"]:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results["per_object"][0].keys()))
            w.writeheader()
            w.writerows(results["per_object"])
    print(f"\nResults written to: {out_json} and {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
