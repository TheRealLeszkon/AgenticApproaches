"""BaseAgent: shared run loop for all four agents.

Responsibilities:
  1. build the stage prompt (system instructions + task + upstream context)
  2. call the model
  3. parse JSON robustly and validate against the stage's Pydantic schema
  4. on failure, issue ONE correction prompt; if that also fails, fail gracefully and
     return a diagnostic dict (never crash the pipeline)

Each concrete agent only supplies: STAGE name, a schema, a system prompt, and a method
that builds the task text from upstream outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from schemas.outputs import STAGE_SCHEMAS
from utils import logging as log
from utils.json_parser import JSONParseError, correction_prompt, parse_json

if TYPE_CHECKING:
    from PIL.Image import Image
    from models.base import VisionModel


class AgentResult:
    """Wraps a stage output plus diagnostics for run artifacts."""

    def __init__(self, stage: str, data: dict[str, Any], validated: BaseModel | None, raw: str, ok: bool):
        self.stage = stage
        self.data = data          # dict form (validated model_dump, or diagnostic)
        self.validated = validated  # pydantic instance or None
        self.raw = raw            # raw model text (last attempt)
        self.ok = ok

    def to_dict(self) -> dict[str, Any]:
        return self.data


class BaseAgent:
    STAGE: str = "base"
    SYSTEM: str = ""
    SCHEMA_HINT: str = "{}"  # short shape shown to the model in the correction prompt

    def __init__(self, model: "VisionModel"):
        self.model = model
        self.schema = STAGE_SCHEMAS.get(self.STAGE)

    # ---- to be overridden ------------------------------------------------- #
    def build_task(self, **ctx: Any) -> str:
        """Return the task portion of the prompt from upstream context."""
        raise NotImplementedError

    def banner(self) -> None:
        log.stage(self.STAGE.capitalize())

    # ---- shared run loop -------------------------------------------------- #
    def _prompt(self, task: str) -> str:
        return f"{self.SYSTEM.strip()}\n\n{task.strip()}"

    def _validate(self, data: dict) -> BaseModel | None:
        if self.schema is None:
            return None
        return self.schema.model_validate(data)

    def run(self, images: list["Image"], **ctx: Any) -> AgentResult:
        self.banner()
        task = self.build_task(**ctx)
        prompt = self._prompt(task)

        raw = self.model.generate(images, prompt)
        try:
            data = parse_json(raw)
            validated = self._validate(data)
            model_dump = validated.model_dump() if validated is not None else data
            log.info("  ok")
            return AgentResult(self.STAGE, model_dump, validated, raw, ok=True)
        except (JSONParseError, ValidationError) as first_err:
            log.warn(f"parse/validate failed ({type(first_err).__name__}); retrying with correction prompt")

        # one correction attempt
        corr = correction_prompt(self.SCHEMA_HINT, raw)
        raw2 = self.model.generate(images, f"{self.SYSTEM.strip()}\n\n{corr}")
        try:
            data = parse_json(raw2)
            validated = self._validate(data)
            model_dump = validated.model_dump() if validated is not None else data
            log.info("  ok (after correction)")
            return AgentResult(self.STAGE, model_dump, validated, raw2, ok=True)
        except (JSONParseError, ValidationError) as second_err:
            log.error(f"stage '{self.STAGE}' produced no valid JSON: {second_err}")
            diag = {
                "_error": f"{type(second_err).__name__}: {second_err}",
                "_raw_first_attempt": raw,
                "_raw_second_attempt": raw2,
            }
            return AgentResult(self.STAGE, diag, None, raw2, ok=False)
