import numpy as np
import OpenImageIO as oiio
import pytest
from fastapi.testclient import TestClient

import open_test_patterns.patterns as P
from open_test_patterns.api.app import app
from open_test_patterns.imageio import SUPPORTED_FORMATS, resolve_compression, write_image

client = TestClient(app)

EXR_IDS = ("none", "zip", "piz")


@pytest.fixture(scope="module")
def result():
    # A pattern with both flat areas and a ramp, so the schemes do real work.
    return P.get_pattern("smpte-rp219-2-2016").render(256, 144, {})


def _read(path):
    src = oiio.ImageInput.open(str(path))
    assert src is not None, oiio.geterror()
    try:
        compression = src.spec().get_string_attribute("compression")
        pixels = src.read_image(format="float")
    finally:
        src.close()
    return compression, np.asarray(pixels)


def test_exr_advertises_three_schemes():
    assert [c.id for c in SUPPORTED_FORMATS["exr"].compressions] == list(EXR_IDS)
    assert SUPPORTED_FORMATS["exr"].default_compression == "zip"
    assert all(c.lossless for c in SUPPORTED_FORMATS["exr"].compressions)


@pytest.mark.parametrize("compression", EXR_IDS)
@pytest.mark.parametrize("bit_depth", (16, 32))
def test_requested_compression_is_written(result, tmp_path, compression, bit_depth):
    path = tmp_path / f"{compression}_{bit_depth}.exr"
    write_image(result, str(path), "exr", bit_depth, compression)
    written, _ = _read(path)
    assert written == compression


def test_default_is_zip(result, tmp_path):
    path = tmp_path / "default.exr"
    write_image(result, str(path), "exr")
    written, _ = _read(path)
    assert written == "zip"


def test_all_schemes_are_lossless(result, tmp_path):
    """Every offered scheme must preserve code values exactly.

    This is the property that matters for test patterns: changing compression
    may change file size but must never change a pixel.
    """
    reference = None
    for compression in EXR_IDS:
        path = tmp_path / f"lossless_{compression}.exr"
        write_image(result, str(path), "exr", 32, compression)
        _, pixels = _read(path)
        if reference is None:
            reference = pixels
        else:
            assert np.array_equal(pixels, reference), f"{compression} altered pixels"
    assert reference is not None
    assert np.allclose(reference, result.image, atol=1e-6)


def test_uncompressed_is_larger_than_compressed(result, tmp_path):
    sizes = {}
    for compression in EXR_IDS:
        path = tmp_path / f"size_{compression}.exr"
        write_image(result, str(path), "exr", 16, compression)
        sizes[compression] = path.stat().st_size
    assert sizes["none"] > sizes["zip"]
    assert sizes["none"] > sizes["piz"]


def test_unknown_compression_is_rejected(result, tmp_path):
    # OpenImageIO would silently fall back to zip, so this must fail loudly.
    with pytest.raises(ValueError, match="supports compressions"):
        write_image(result, str(tmp_path / "bad.exr"), "exr", 16, "dwaa")


def test_compression_rejected_for_formats_without_it(result, tmp_path):
    with pytest.raises(ValueError, match="does not support a compression choice"):
        write_image(result, str(tmp_path / "bad.png"), "png", 16, "zip")


def test_resolve_compression_defaults():
    assert resolve_compression(SUPPORTED_FORMATS["exr"], None) == "zip"
    assert resolve_compression(SUPPORTED_FORMATS["exr"], "piz") == "piz"
    assert resolve_compression(SUPPORTED_FORMATS["png"], None) is None


def test_formats_endpoint_exposes_compressions():
    formats = {f["id"]: f for f in client.get("/api/formats").json()}
    assert [c["id"] for c in formats["exr"]["compressions"]] == list(EXR_IDS)
    assert formats["exr"]["default_compression"] == "zip"
    assert formats["png"]["compressions"] == []
    assert formats["png"]["default_compression"] is None


@pytest.mark.parametrize("compression", EXR_IDS)
def test_render_endpoint_accepts_compression(compression):
    r = client.post(
        "/api/render",
        json={
            "pattern_id": "smpte-rp219-2-2016",
            "width": 128,
            "height": 72,
            "params": {},
            "format": "exr",
            "bit_depth": 16,
            "compression": compression,
        },
    )
    assert r.status_code == 200
    assert r.content[:4] == b"\x76\x2f\x31\x01"  # OpenEXR magic


def test_render_endpoint_rejects_bad_compression():
    r = client.post(
        "/api/render",
        json={
            "pattern_id": "smpte-rp219-2-2016",
            "width": 64,
            "height": 36,
            "params": {},
            "format": "exr",
            "compression": "b44",
        },
    )
    assert r.status_code == 400


@pytest.mark.parametrize("pattern", P.all_patterns(), ids=lambda p: p.id)
@pytest.mark.parametrize("compression", EXR_IDS)
def test_every_pattern_writes_with_every_scheme(pattern, compression, tmp_path):
    path = tmp_path / f"{pattern.id}_{compression}.exr"
    write_image(pattern.render(64, 36, {}), str(path), "exr", 16, compression)
    written, _ = _read(path)
    assert written == compression
