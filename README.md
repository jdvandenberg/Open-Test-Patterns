# Open Test Patterns

Modern, extensible generator of color-accurate **test patterns** for display
calibration, HDR mastering QC, and wide-gamut verification.

This is a ground-up rewrite of a long-standing collection of scripts used to
author reference charts (SMPTE color bars, ColorChecker, PQ gray steps, zone
plates, geometry/alignment grids, and more). It is organized as a small Python
web application so patterns can be previewed in the browser and exported to
production formats (OpenEXR, DPX, TIFF, PNG).

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

## License

Code is provided under the Apache-2.0 license. Generated assets contributed to
DPEL are distributed under the
[ASWF Digital Assets License](https://dpel.aswf.io/).
