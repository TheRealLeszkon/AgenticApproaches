"""Judge agent — weigh evidence and produce the final, honest answer."""

from __future__ import annotations

import json

from agents.base import BaseAgent

_SCHEMA_HINT = """{
  "object": str,
  "dimensions": {"width_cm": float, "height_cm": float, "depth_cm": float},
  "ranges": {"width_cm": [min, max], "height_cm": [min, max], "depth_cm": [min, max]},
  "confidence": {"overall": float, "width": float, "height": float, "depth": float},
  "evidence_quality": "low|moderate|high",
  "assumptions": [str],
  "warnings": [str]
}"""


class Judge(BaseAgent):
    STAGE = "judge"
    SCHEMA_HINT = _SCHEMA_HINT
    SYSTEM = """You are the JUDGE in a multi-agent object-dimension system. You see the
Observer report, the Estimator output, and the Critic's objections. Produce the FINAL
answer.

Do NOT blindly average numbers. Instead weigh the evidence:
- Favor dimensions backed by reliable scale references or consistent multi-view evidence.
- Take the Critic's valid objections seriously; widen ranges or lower confidence where the
  Critic exposed weak evidence.
- Keep the point estimate honest and give a plausible [min, max] RANGE for each dimension.
  Poorly constrained dimensions (often depth) should have WIDER ranges and lower confidence.

NEVER hallucinate precision. Prefer round numbers and honest ranges (e.g. 45-50 cm) over
impressive-looking exact figures. It is better to say a dimension is uncertain than to fake
confidence.

Populate "assumptions" with the key assumptions the answer depends on, and "warnings" with
anything the user should be cautious about (e.g. "depth is far less certain than width").

Respond with ONLY a single JSON object in this shape:
""" + _SCHEMA_HINT

    def build_task(
        self,
        num_images: int = 0,
        observation: dict | None = None,
        estimate: dict | None = None,
        critique: dict | None = None,
        **_ctx,
    ) -> str:
        obs = json.dumps(observation or {}, indent=2)
        est = json.dumps(estimate or {}, indent=2)
        crit = json.dumps(critique or {}, indent=2)
        return (
            f"You are given {num_images} image(s) of the object.\n\n"
            f"Observer report:\n{obs}\n\n"
            f"Estimator output:\n{est}\n\n"
            f"Critic output:\n{crit}\n\n"
            "Now aggregate the evidence and produce the final JSON."
        )
