from __future__ import annotations

import argparse
from pathlib import Path

from .render import render_index


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="viz", description="Visualization utilities")
    sub = p.add_subparsers(dest="cmd")

    p_render = sub.add_parser("render", help="Render static dashboard to index.html")
    p_render.add_argument("--log-dir", type=str, default=".viz", help="Directory with logs (scalars.csv, etc.)")
    p_render.add_argument("--title", type=str, default=None)

    args = p.parse_args(argv)

    if args.cmd == "render":
        out = render_index(args.log_dir, title=args.title)
        print(str(out))
        return

    p.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()


