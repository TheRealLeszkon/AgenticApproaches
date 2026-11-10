"""Pydantic schemas for each agent stage.

These are intentionally *lenient*: fields are Optional and unknown extra keys are
allowed. Validation here is for structure/diagnostics and to give downstream agents
a predictable shape to read — NOT to reject imperfect-but-usable model output. VLMs
(especially small/quantized ones) vary a lot in what they emit.

Dimensions are always in centimetres.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    # allow extra keys so we never drop model-provided context, and never crash on it
    model_config = ConfigDict(extra="allow")


# --------------------------------------------------------------------------- #
# Observer
# --------------------------------------------------------------------------- #
class ScaleReference(_Base):
    description: str = ""
    reliability: Optional[float] = None


class Geometry(_Base):
    shape: Optional[str] = None
    proportions: dict[str, Any] = Field(default_factory=dict)


class ImageQuality(_Base):
    overall: Optional[float] = None
    useful_images: list[int] = Field(default_factory=list)


class Observation(_Base):
    object: Optional[str] = None
    object_confidence: Optional[float] = None
    same_object_across_images: Optional[bool] = None
    visible_faces: list[str] = Field(default_factory=list)
    geometry: Geometry = Field(default_factory=Geometry)
    scale_references: list[ScaleReference] = Field(default_factory=list)
    image_quality: ImageQuality = Field(default_factory=ImageQuality)
    constrained_dimensions: list[str] = Field(default_factory=list)
    uncertain_dimensions: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #
class Dimensions(_Base):
    width_cm: Optional[float] = None
    height_cm: Optional[float] = None
    depth_cm: Optional[float] = None


class Estimate(_Base):
    dimensions: Dimensions = Field(default_factory=Dimensions)
    # evidence supporting each dimension (list of short strings)
    evidence: dict[str, list[str]] = Field(default_factory=dict)
    # per-dimension epistemic basis: "observed" | "inferred" | "assumed"
    basis: dict[str, str] = Field(default_factory=dict)
    # per-dimension confidence in [0, 1]
    confidence: dict[str, float] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Critic
# --------------------------------------------------------------------------- #
class Problem(_Base):
    dimension: Optional[str] = None
    issue: str = ""


class Range(_Base):
    min: Optional[float] = None
    max: Optional[float] = None


class Critique(_Base):
    validity: Optional[str] = None  # e.g. "plausible" | "questionable" | "unreliable"
    problems: list[Problem] = Field(default_factory=list)
    # suggested min/max per dimension, keyed e.g. "depth_cm"
    suggested_adjustment: dict[str, Range] = Field(default_factory=dict)
    confidence: Optional[float] = None


# --------------------------------------------------------------------------- #
# Judge (final)
# --------------------------------------------------------------------------- #
class JudgeResult(_Base):
    object: Optional[str] = None
    dimensions: Dimensions = Field(default_factory=Dimensions)
    # [min, max] per dimension, keyed e.g. "width_cm"
    ranges: dict[str, list[float]] = Field(default_factory=dict)
    confidence: dict[str, float] = Field(default_factory=dict)
    evidence_quality: Optional[str] = None  # "low" | "moderate" | "high"
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# Map stage name -> schema, used by the agent base class.
STAGE_SCHEMAS = {
    "observer": Observation,
    "estimator": Estimate,
    "critic": Critique,
    "judge": JudgeResult,
}
