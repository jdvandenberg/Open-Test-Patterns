"""FastAPI service exposing the pattern catalog, previews, and downloads."""

from __future__ import annotations

import os
import tempfile

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

from .. import __version__
from ..color import colorspaces, transfer
from ..imageio import SUPPORTED_FORMATS, render_preview_png, write_image
from ..patterns import all_patterns, get_pattern
from ..schemas import (
    ColorSpaceModel,
    CompressionModel,
    FormatModel,
    ParameterModel,
    PatternModel,
    PreviewRequest,
    RenderRequest,
    TransferModel,
)

app = FastAPI(
    title="Open Test Patterns",
    version=__version__,
    description="Generate color-accurate test patterns for display calibration and HDR QC.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cap the working resolution for previews; the result is downscaled anyway.
_PREVIEW_MAX_DIM = 2048


def _pattern_model(pattern) -> PatternModel:
    return PatternModel(
        id=pattern.id,
        name=pattern.name,
        category=pattern.category,
        description=pattern.description,
        parameters=[ParameterModel.from_parameter(p) for p in pattern.parameters],
    )


def _preview_dimensions(width: int, height: int) -> tuple[int, int]:
    longest = max(width, height)
    if longest <= _PREVIEW_MAX_DIM:
        return width, height
    scale = _PREVIEW_MAX_DIM / float(longest)
    return max(1, round(width * scale)), max(1, round(height * scale))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/api/patterns", response_model=list[PatternModel])
def list_patterns() -> list[PatternModel]:
    return [_pattern_model(p) for p in all_patterns()]


@app.get("/api/patterns/{pattern_id}", response_model=PatternModel)
def get_pattern_detail(pattern_id: str) -> PatternModel:
    try:
        return _pattern_model(get_pattern(pattern_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/colorspaces", response_model=list[ColorSpaceModel])
def list_colorspaces() -> list[ColorSpaceModel]:
    return [
        ColorSpaceModel(id=cs.id, name=cs.name, primaries=cs.primaries, whitepoint=cs.whitepoint)
        for cs in colorspaces.list_colorspaces()
    ]


@app.get("/api/transfers", response_model=list[TransferModel])
def list_transfers() -> list[TransferModel]:
    return [
        TransferModel(
            id=tf.id,
            name=tf.name,
            description=tf.description,
            is_absolute=tf.is_absolute,
            is_bypass=tf.is_bypass,
        )
        for tf in transfer.list_transfer_functions()
    ]


@app.get("/api/formats", response_model=list[FormatModel])
def list_formats() -> list[FormatModel]:
    return [
        FormatModel(
            id=f.id,
            label=f.label,
            extension=f.extension,
            default_bit_depth=f.default_bit_depth,
            allowed_bit_depths=list(f.allowed_bit_depths),
            compressions=[
                CompressionModel(id=c.id, label=c.label, lossless=c.lossless)
                for c in f.compressions
            ],
            default_compression=f.default_compression,
        )
        for f in SUPPORTED_FORMATS.values()
    ]


@app.post("/api/preview")
def preview(req: PreviewRequest) -> Response:
    try:
        pattern = get_pattern(req.pattern_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    width, height = _preview_dimensions(req.width, req.height)
    try:
        result = pattern.render(width, height, req.params)
        png = render_preview_png(result)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=png, media_type="image/png")


@app.post("/api/render")
def render(req: RenderRequest) -> StreamingResponse:
    try:
        pattern = get_pattern(req.pattern_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if req.format not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unknown format {req.format!r}")

    fmt = SUPPORTED_FORMATS[req.format]
    try:
        result = pattern.render(req.width, req.height, req.params)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    fd, tmp_path = tempfile.mkstemp(suffix=f".{fmt.extension}")
    os.close(fd)
    try:
        write_image(result, tmp_path, req.format, req.bit_depth, req.compression)
    except (ValueError, RuntimeError) as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    filename = f"{req.pattern_id}_{req.width}x{req.height}.{fmt.extension}"
    return FileResponse(
        tmp_path,
        media_type=fmt.mime,
        filename=filename,
        background=BackgroundTask(lambda: os.path.exists(tmp_path) and os.remove(tmp_path)),
    )
