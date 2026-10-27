"""Command-line entry points: API server, worker, model management, cleanup, migrations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def api(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stemforge-api")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run("stemforge.api.main:app", host=args.host, port=args.port, reload=args.reload,
                log_config=None, proxy_headers=True)


def worker(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stemforge-worker")
    parser.add_argument("--concurrency", type=int, default=None, help="worker processes (default from settings)")
    args = parser.parse_args(argv)
    from .worker import main

    main(args.concurrency)


def models(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stemforge-models", description="Download and verify pinned model weights")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("names", nargs="*", help="models to fetch (default: all default models)")
    d.add_argument("--source-file", default=None,
                   help="use an already downloaded upstream file (one model only); it is verified like a download")
    sub.add_parser("verify")
    sub.add_parser("list")
    args = parser.parse_args(argv)
    from . import model_registry as reg

    if args.cmd == "list":
        for name, spec in reg.manifest().items():
            st = reg.status(name)
            print(f"{name:14s} {'installed' if st['installed'] else 'missing  '} {st['size_bytes'] / 1e6:7.1f} MB  "
                  f"{'stems=' + ','.join(spec['sources']) if spec['sources'] else spec.get('purpose', '')}")
            print(f"{'':14s} license: {spec['license']}")
        return
    if args.cmd == "verify":
        failed = False
        for name in reg.manifest():
            try:
                reg.verify(name)
                print(f"{name}: OK")
            except reg.ModelError as exc:
                print(f"{name}: {exc}")
                failed = failed or reg.manifest()[name].get("default", False)
        sys.exit(1 if failed else 0)
    names = args.names or [n for n, s in reg.manifest().items() if s.get("default")]
    if args.source_file and len(names) != 1:
        parser.error("--source-file needs exactly one model name")
    for name in names:
        reg.download(name, source_file=Path(args.source_file) if args.source_file else None)
        reg.verify(name)
        print(f"{name}: installed and verified")


def cleanup(argv: list[str] | None = None) -> None:
    from .db import session_scope
    from .logging_setup import configure_logging
    from .migrate import upgrade
    from .services import cleanup_expired

    configure_logging()
    upgrade()
    with session_scope() as db:
        n = cleanup_expired(db)
    print(f"deleted {n} expired project(s)")


def migrate(argv: list[str] | None = None) -> None:
    from .migrate import upgrade

    upgrade()
    print("database is up to date")
