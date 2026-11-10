"""Critic agent — actively try to DISPROVE the estimate."""

from __future__ import annotations

import json

from agents.base import BaseAgent

_SCHEMA_HINT = """{
  "validity": "plausible|questionable|unreliable",
  "problems": [{"dimension": "width|height|depth", "issue": str}],
  "suggested_adjustment": {"depth_cm": {"min": float, "max": float}},
  "confidence": float(0-1)
}"""


class Critic(BaseAgent):
    STAGE = "critic"
    SCHEMA_HINT = _SCHEMA_HINT
    SYSTEM = """You are the CRITIC in a multi-agent object-dimension system. Your explicit
job is to TRY TO PROVE THE ESTIMATE WRONG. Be skeptical and adversarial — assume the
Estimator may have made mistakes, and hunt for them.

Look hard for:
- perspective distortion and foreshortening making sizes look wrong
- misleading, misidentified, or fabricated scale references
- incorrect object identification
- hidden or occluded depth (the back of the object is usually unseen)
- camera-angle problems and cropping
- dimensions that are inconsistent between the different images
- over-reliance on "typical" object dimensions dressed up as observation
- physically impossible or implausible proportions
- contradictory evidence

Where you find a problem, propose a corrected plausible RANGE (min/max in cm) for the
affected dimension. It is good to disagree with the Estimator when the evidence is weak.

Respond with ONLY a single JSON object in this shape:
""" + _SCHEMA_HINT

    def build_task(
        self,
        num_images: int = 0,
        observation: dict | None = None,
        estimate: dict | None = None,
        **_ctx,
    ) -> str:
        obs = json.dumps(observation or {}, indent=2)
        est = json.dumps(estimate or {}, indent=2)
        return (
            f"You are given {num_images} image(s) of the object.\n\n"
            f"Observer report:\n{obs}\n\n"
            f"Estimator output:\n{est}\n\n"
            "Now critique the estimate and produce the critique JSON."
        )
