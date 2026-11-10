"""Estimator agent — estimate W/H/D, distinguishing observed / inferred / assumed."""

from __future__ import annotations

import json

from agents.base import BaseAgent

_SCHEMA_HINT = """{
  "dimensions": {"width_cm": float, "height_cm": float, "depth_cm": float},
  "evidence": {"width": [str], "height": [str], "depth": [str]},
  "basis": {"width": "observed|inferred|assumed", "height": "...", "depth": "..."},
  "confidence": {"width": float(0-1), "height": float(0-1), "depth": float(0-1)},
  "assumptions": [str]
}"""


class Estimator(BaseAgent):
    STAGE = "estimator"
    SCHEMA_HINT = _SCHEMA_HINT
    SYSTEM = """You are the ESTIMATOR in a multi-agent object-dimension system. Using the
images AND the Observer's report, estimate the object's real-world WIDTH, HEIGHT and DEPTH
in centimetres.

Reason from: visual proportions, perspective across multiple viewpoints, the recognized
object category and its typical dimensions, and any genuine scale references the Observer
found. Relate unknown sizes to known-size objects when possible.

For EACH dimension you MUST label its basis:
- "observed": directly readable from a reliable in-image scale reference
- "inferred": reasonably deduced from visual evidence (proportions, perspective, category)
- "assumed":  based mainly on typical/prior dimensions for this kind of object

HONESTY OVER PRECISION — this is the most important rule:
- You have NOT physically measured anything. Do not pretend to.
- Do NOT output false precision like 47.283 cm. Round to sensible whole centimetres.
- If a dimension is poorly constrained (often depth), give a lower confidence and rely on
  "assumed", and list the assumption in "assumptions".

Respond with ONLY a single JSON object in this shape:
""" + _SCHEMA_HINT

    def build_task(self, num_images: int = 0, observation: dict | None = None, **_ctx) -> str:
        obs = json.dumps(observation or {}, indent=2)
        return (
            f"You are given {num_images} image(s) of the object.\n\n"
            f"Observer report:\n{obs}\n\n"
            "Produce the estimate JSON now."
        )
