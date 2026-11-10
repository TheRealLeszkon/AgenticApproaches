"""Pipeline orchestration: Observer -> Estimator -> Critic -> Judge.

Plain Python — no framework. Loads one model, runs the four agents in sequence passing
each stage's output forward, and writes intermediate + final JSON to a per-run directory.
Shared by main.py and benchmark.py.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from typing import TYPE_CHECKING, Any

from agents.critic import Critic
from agents.estimator import Estimator
from agents.judge import Judge
from agents.observer import Observer
from utils import logging as log

if TYPE_CHECKING:
    from PIL.Image import Image
    from models.base import VisionModel


def _write_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_pipeline(
    model: "VisionModel",
    images: list["Image"],
    image_paths: list[str] | None = None,
    output_dir: str = "runs",
    run_name: str | None = None,
) -> dict[str, Any]:
    """Run the full 4-agent pipeline. Returns the final judge dict.

    Also writes {observer,estimator,critic,judge,final}.json under
    ``output_dir/<run_name or timestamp>/``.
    """
    num_images = len(images)
    log.info(f"Loaded {num_images} image(s) for model '{getattr(model, 'name', '?')}'.")

    run_name = run_name or _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    observer = Observer(model)
    estimator = Estimator(model)
    critic = Critic(model)
    judge = Judge(model)

    obs_res = observer.run(images, num_images=num_images)
    _write_json(os.path.join(run_dir, "observer.json"), obs_res.to_dict())

    est_res = estimator.run(images, num_images=num_images, observation=obs_res.to_dict())
    _write_json(os.path.join(run_dir, "estimator.json"), est_res.to_dict())

    crit_res = critic.run(
        images, num_images=num_images, observation=obs_res.to_dict(), estimate=est_res.to_dict()
    )
    _write_json(os.path.join(run_dir, "critic.json"), crit_res.to_dict())

    judge_res = judge.run(
        images,
        num_images=num_images,
        observation=obs_res.to_dict(),
        estimate=est_res.to_dict(),
        critique=crit_res.to_dict(),
    )

    final = judge_res.to_dict()
    # attach lightweight provenance so a run folder is self-describing
    final_with_meta = dict(final)
    final_with_meta["_meta"] = {
        "model": getattr(model, "name", "?"),
        "num_images": num_images,
        "image_paths": image_paths or [],
        "run_dir": run_dir,
        "stage_ok": {
            "observer": obs_res.ok,
            "estimator": est_res.ok,
            "critic": crit_res.ok,
            "judge": judge_res.ok,
        },
    }
    _write_json(os.path.join(run_dir, "judge.json"), final)
    _write_json(os.path.join(run_dir, "final.json"), final_with_meta)

    log.final_summary(final)
    log.info(f"\nRun artifacts written to: {run_dir}")
    return final_with_meta
