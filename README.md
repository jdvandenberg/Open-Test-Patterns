# Open Test Patterns

Modern, extensible generator of color-accurate **test patterns** for display
calibration, HDR mastering QC, and wide-gamut verification.

This is a ground-up rewrite of a long-standing collection of scripts used to
author reference charts (SMPTE color bars, ColorChecker, PQ gray steps, zone
plates, geometry/alignment grids, and more). It is organized as a small Python
web application so patterns can be previewed in the browser and exported to
production formats (OpenEXR, DPX, TIFF, PNG).

The web UI is available at
[https://open-test-patterns.onrender.com](https://open-test-patterns.onrender.com).

## Motivation

Most published test patterns exist as one of two things: a description and
table of values in a standards document (SMPTE, ITU-R, ARIB), or a
vendor-authored artifact (a Blu-ray disc, a proprietary hardware generator).
The Open Test Patterns initiative aims to close that gap for a broader set of
patterns, formats, and colorimetries.

Why a generator rather than a library of pre-rendered files:

- **Resolution, frame rate, file format, and color space agnostic.** A
  generator produces exactly what a given pipeline needs (e.g., an 8K color
  bars chart encoded in Rec.2100 PQ) instead of shipping hundreds of pre-baked
  permutations and hoping the one you need is in the set.
- **Easily extensible.** Adding a new pattern, colorimetry, or transfer
  function is a matter of writing a new generator against a typed parameter
  schema, not re-authoring a disc image or re-running a closed toolchain.
- **Lightweight.** Patterns render on-demand, on-prem or in CI, without needing
  to distribute or archive large pre-rendered asset sets.
- **Reliable, inspectable targets.** Code values are hard-coded where the
  governing standard specifies them (e.g. exact 10-/12-bit Y'CbCr integers from
  SMPTE RP 219-2 Annex A/B), rather than derived from floating-point math and
  rounded at render time. That way, the same input always produces the same,
  independently verifiable output.
- **Built on trusted, existing open-source foundations.** Color science,
  transforms, and I/O are delegated to established libraries (colour-science,
  OpenColorIO-compatible transfer functions, OpenImageIO) rather than
  reimplemented, so correctness rests on widely-used, independently audited
  code rather than a bespoke implementation.

## Architecture

```
Open Test Patterns/
├── backend/                     # Python package + FastAPI service
│   └── open_test_patterns/
│       ├── color/               # color engine (colour-science based)
│       ├── patterns/            # pattern generators + registry
│       ├── imageio/             # OpenImageIO writers + preview encoding
│       ├── api/                 # FastAPI application
│       ├── schemas.py           # shared pydantic models
│       └── cli.py               # command-line entry point
├── frontend/                    # React + Vite single-page app
├── tests/                       # pytest suite
└── pyproject.toml
```

### Core concepts

- **Color engine** (`open_test_patterns.color`) wraps
  [colour-science](https://www.colour-science.org/) for color-space matrices,
  chromatic adaptation, and transfer functions (PQ / ST 2084, HLG, gamma,
  legal/full range).
- **Patterns** (`open_test_patterns.patterns`) are self-describing generators.
  Each declares typed parameters and returns a floating-point image plus a
  `SignalFormat` describing its color space, transfer function, and range so it
  can be written and tagged correctly.
- **Image I/O** (`open_test_patterns.imageio`) uses
  [OpenImageIO](https://openimageio.readthedocs.io/) for EXR/DPX/TIFF/PNG output
  and to produce lightweight sRGB PNG previews for the browser.

## Getting started

### Backend

Requires Python 3.10+ (3.12 recommended). Using [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Run the API server:

```bash
otp serve            # or: uvicorn open_test_patterns.api.app:app --reload
```

Generate a pattern from the command line:

```bash
otp list
otp render smpte-rp219-2-2016 --width 3840 --height 2160 -o bars.exr
```

OpenEXR output can use any of three lossless compression schemes —
`none` (uncompressed), `zip` (the default), or `piz`:

```bash
otp render smpte-rp219-2-2016 -o bars.exr --compression piz
```

### Frontend

Requires Node 18+.

```bash
cd frontend
npm install
npm run dev
```

The dev server proxies API requests to the backend on port 8000.

### Deploy (Docker / Render)

The repo has a `Dockerfile` that builds the frontend and runs FastAPI as a
single service — API at `/api/*`, the UI at `/`. Render will set `PORT` for
you.

```bash
docker build -t open-test-patterns .
docker run --rm -p 8000:8000 open-test-patterns
```

On [Render](https://render.com): New Web Service → this GitHub repo →
**Docker**. Use `/api/health` as the health-check path. OpenImageIO is not
tiny; a free instance may run out of memory on 4K/8K renders.

## License

Code is provided under the Apache-2.0 license. Generated assets contributed to
DPEL are distributed under the
[ASWF Digital Assets License](https://dpel.aswf.io/).
