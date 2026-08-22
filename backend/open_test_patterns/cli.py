"""Command-line interface for Open Test Patterns."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .imageio import SUPPORTED_FORMATS, resolve_compression, write_image
from .patterns import all_patterns, get_pattern
from .patterns.base import ParamType


def _cmd_list(args: argparse.Namespace) -> int:
    patterns = all_patterns()
    if args.json:
        payload = [
            {
                "id": p.id,
                "name": p.name,
                "category": p.category,
                "description": p.description,
            }
            for p in patterns
        ]
        print(json.dumps(payload, indent=2))
        return 0
    category = None
    for p in patterns:
        if p.category != category:
            category = p.category
            print(f"\n{category}")
        print(f"  {p.id:<22} {p.name}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    try:
        pattern = get_pattern(args.pattern)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"{pattern.name} ({pattern.id})\n{pattern.description}\n")
    print("Parameters:")
    for p in pattern.parameters:
        extra = ""
        if p.choices:
            extra = " {" + ", ".join(c.value for c in p.choices) + "}"
        elif p.type in (ParamType.INT, ParamType.FLOAT):
            extra = f" [{p.minimum}, {p.maximum}]"
        print(f"  {p.name:<18} {p.type.value:<7} default={p.default!r}{extra}")
    return 0


def _parse_param(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Expected key=value, got {raw!r}")
    key, value = raw.split("=", 1)
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def _cmd_render(args: argparse.Namespace) -> int:
    try:
        pattern = get_pattern(args.pattern)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1
    params = dict(_parse_param(p) for p in args.param)
    result = pattern.render(args.width, args.height, params)

    ext = args.output.rsplit(".", 1)[-1].lower()
    format_id = next((f.id for f in SUPPORTED_FORMATS.values() if f.extension == ext), None)
    if format_id is None:
        format_id = ext if ext in SUPPORTED_FORMATS else "png"
    try:
        write_image(result, args.output, format_id, args.bit_depth, args.compression)
    except (ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1

    fmt = SUPPORTED_FORMATS[format_id]
    detail = f"{args.width}x{args.height}, {format_id}"
    compression = resolve_compression(fmt, args.compression)
    if compression is not None:
        detail += f", {compression}"
    print(f"Wrote {args.output} ({detail})")
    return 0


def _port_in_use(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return False
        except OSError:
            return True


def _cmd_serve(args: argparse.Namespace) -> int:
    import os
    from pathlib import Path

    import uvicorn

    # Ensure the package is importable even if the editable .pth was skipped
    # (modern Python ignores underscore-prefixed hatch .pth files).
    backend_dir = Path(__file__).resolve().parents[1]
    pythonpath = os.environ.get("PYTHONPATH", "")
    parts = [p for p in pythonpath.split(os.pathsep) if p]
    if str(backend_dir) not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(backend_dir), *parts])

    if _port_in_use(args.host, args.port):
        print(
            f"Port {args.port} on {args.host} is already in use.\n"
            f"Another server is probably still running. You can either:\n"
            f"  - stop it:   kill $(lsof -nP -tiTCP:{args.port} -sTCP:LISTEN)\n"
            f"  - use another port:  otp serve --port {args.port + 1}",
            file=sys.stderr,
        )
        return 1

    uvicorn.run(
        "open_test_patterns.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="otp", description="Open Test Patterns")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List available patterns")
    p_list.add_argument("--json", action="store_true", help="Emit JSON")
    p_list.set_defaults(func=_cmd_list)

    p_info = sub.add_parser("info", help="Show a pattern's parameters")
    p_info.add_argument("pattern")
    p_info.set_defaults(func=_cmd_info)

    p_render = sub.add_parser("render", help="Render a pattern to a file")
    p_render.add_argument("pattern")
    p_render.add_argument(
        "-o", "--output", required=True, help="Output file (extension picks format)"
    )
    p_render.add_argument("--width", type=int, default=1920)
    p_render.add_argument("--height", type=int, default=1080)
    p_render.add_argument("--bit-depth", type=int, default=None)
    p_render.add_argument(
        "--compression",
        default=None,
        choices=sorted({c.id for f in SUPPORTED_FORMATS.values() for c in f.compressions}),
        help="Compression scheme (OpenEXR only; default zip)",
    )
    p_render.add_argument("--param", action="append", default=[], help="key=value (value is JSON)")
    p_render.set_defaults(func=_cmd_render)

    p_serve = sub.add_parser("serve", help="Run the API server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
