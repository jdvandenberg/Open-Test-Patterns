"""Pydantic models shared by the API layer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .patterns.base import Parameter


class ChoiceModel(BaseModel):
    value: str
    label: str


class DisabledWhenModel(BaseModel):
    parameter: str
    values: list[str]
    reason: str = ""


class ParameterModel(BaseModel):
    name: str
    label: str
    type: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: list[ChoiceModel] = Field(default_factory=list)
    unit: str | None = None
    description: str = ""
    disabled_when: list[DisabledWhenModel] = Field(default_factory=list)

    @classmethod
    def from_parameter(cls, p: Parameter) -> ParameterModel:
        return cls(
            name=p.name,
            label=p.label,
            type=p.type.value,
            default=p.default,
            minimum=p.minimum,
            maximum=p.maximum,
            step=p.step,
            choices=[ChoiceModel(value=c.value, label=c.label) for c in p.choices],
            unit=p.unit,
            description=p.description,
            disabled_when=[
                DisabledWhenModel(
                    parameter=cond.parameter,
                    values=list(cond.values),
                    reason=cond.reason,
                )
                for cond in p.disabled_conditions()
            ],
        )


class PatternModel(BaseModel):
    id: str
    name: str
    category: str
    description: str
    parameters: list[ParameterModel]
    kind: str = "image"


class ColorSpaceModel(BaseModel):
    id: str
    name: str
    primaries: list[list[float]]
    whitepoint: list[float]


class TransferModel(BaseModel):
    id: str
    name: str
    description: str = ""
    is_absolute: bool = False
    is_bypass: bool = False


class CompressionModel(BaseModel):
    id: str
    label: str
    lossless: bool = True


class FormatModel(BaseModel):
    id: str
    label: str
    extension: str
    default_bit_depth: int
    allowed_bit_depths: list[int]
    compressions: list[CompressionModel] = Field(default_factory=list)
    default_compression: str | None = None


class RenderRequest(BaseModel):
    pattern_id: str
    width: int = Field(default=1920, ge=1, le=16384)
    height: int = Field(default=1080, ge=1, le=16384)
    params: dict[str, Any] = Field(default_factory=dict)
    format: str = "png"
    bit_depth: int | None = None
    compression: str | None = None


class PreviewRequest(BaseModel):
    pattern_id: str
    width: int = Field(default=1920, ge=1, le=16384)
    height: int = Field(default=1080, ge=1, le=16384)
    params: dict[str, Any] = Field(default_factory=dict)


class ToneRequest(BaseModel):
    frequency: float = Field(default=440.0, gt=0, le=20_000)
    duration: float = Field(default=5.0, gt=0, le=60)
    loudness: float = Field(default=-20.0, le=0, ge=-60)
    waveform: Literal["sine", "triangle", "sawtooth", "square", "sweep", "white", "pink"] = "sine"
    frequency_low: float = Field(default=20.0, gt=0, le=20_000)
    frequency_high: float = Field(default=20_000.0, gt=0, le=20_000)
    sample_rate: int = Field(default=44_100)
    bit_depth: Literal["16", "24", "float32"] = "16"

    @field_validator("sample_rate", mode="before")
    @classmethod
    def _sample_rate(cls, value: object) -> int:
        rate = int(str(value))
        if rate not in {44_100, 48_000, 96_000, 192_000}:
            raise ValueError("sample_rate must be 44100, 48000, 96000, or 192000")
        return rate
