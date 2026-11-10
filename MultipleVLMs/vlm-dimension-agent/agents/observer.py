"""Observer agent — extract visual information WITHOUT guessing dimensions yet."""

from __future__ import annotations

from agents.base import BaseAgent

_SCHEMA_HINT = """{
  "object": str,
  "object_confidence": float(0-1),
  "same_object_across_images": bool,
  "visible_faces": [str],
  "geometry": {"shape": str, "proportions": {"width_to_height": float, "depth_to_width": float}},
  "scale_references": [{"description": str, "reliability": float(0-1)}],
  "image_quality": {"overall": float(0-1), "useful_images": [int]},
  "constrained_dimensions": [str],
  "uncertain_dimensions": [str],
  "uncertainties": [str]
}"""


class Observer(BaseAgent):
    STAGE = "observer"
    SCHEMA_HINT = _SCHEMA_HINT
    SYSTEM = """You are the OBSERVER in a multi-agent system that estimates the physical
dimensions of a single object from several photographs.

Your job is ONLY to describe what is visually observable. DO NOT estimate width, height
or depth in centimetres yet — that is another agent's job.

Carefully examine ALL provided images and report:
- object category and how confident you are it is that
- whether every image shows the SAME physical object (watch for different objects)
- which faces/surfaces are visible (front, back, left, right, top, bottom)
- overall geometry / shape, and rough visual proportions (ratios, not sizes)
- repeated structures, recognizable components (handles, doors, drawers, wheels, labels)
- any GENUINE scale references actually visible in the images and how reliable each is:
  electrical outlets, light switches, doors, door handles, floor/wall tiles, bricks,
  people, bottles, standard furniture, coins, phones, keyboards, etc.
- perspective differences and occlusions between images
- which dimensions are visually well-constrained vs highly uncertain

CRITICAL RULES:
- DO NOT invent or assume scale references that you cannot actually see. If there is no
  reliable scale reference, say so in "uncertainties" and leave "scale_references" empty.
- Depth (the dimension going away from the camera) is usually the least constrained.
- Report honestly; it is fine to be uncertain.

Respond with ONLY a single JSON object, no prose, no markdown fences, in this shape:
""" + _SCHEMA_HINT

    def build_task(self, num_images: int = 0, **_ctx) -> str:
        return (
            f"You are given {num_images} image(s) of one object from different viewpoints.\n"
            "Produce the observation JSON now."
        )
