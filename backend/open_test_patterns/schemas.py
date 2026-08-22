"""Pydantic models shared by the API layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

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
    disabled_when: DisabledWhenModel | None = None

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
            disabled_when=(
                DisabledWhenModel(
                    parameter=p.disabled_when.parameter,
                    values=list(p.disabled_when.values),
                    reason=p.disabled_when.reason,
                )
                if p.disabled_when
                else None
            ),
        )


class PatternModel(BaseModel):
    id: str
    name: str
    category: str
    description: str
    parameters: list[ParameterModel]


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
