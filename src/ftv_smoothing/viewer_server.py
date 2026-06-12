"""Serve generated Forge3D viewer files over localhost."""

from __future__ import annotations

import argparse
import logging
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class NoCacheHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Serve regenerated model files without stale browser cache entries."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def build_parser() -> argparse.ArgumentParser:
    """Create the local viewer server CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Serve a generated viewer until interrupted."""

    args = build_parser().parse_args(argv)
    directory = args.directory.resolve()
    if not (directory / "index.html").exists():
        raise SystemExit(f"Viewer index not found: {directory / 'index.html'}")
    handler = partial(NoCacheHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    LOGGER.warning("Serving Forge3D viewer at http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
