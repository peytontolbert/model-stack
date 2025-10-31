from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .card import write_model_card
from .lineage import lineage_from_training_metadata, write_lineage_graph
from .receipt import generate_reproducibility_receipt
from .sbom import create_spdx_sbom
from .signature import sign_file_ed25519, write_checksums_file


def _read_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cmd_card(args: argparse.Namespace) -> int:
    meta = _read_json(args.metadata)
    sboms = args.sbom or []
    out = write_model_card(args.artifact, meta, sbom_paths=sboms, output_path=args.out)
    print(str(out))
    return 0


def cmd_sbom(args: argparse.Namespace) -> int:
    out = create_spdx_sbom(args.out, base_name=args.name)
    print(str(out))
    return 0


def cmd_sign(args: argparse.Namespace) -> int:
    checksums = write_checksums_file(args.files, args.out)
    print(str(checksums))
    if args.key:
        for f in args.files:
            sigp = sign_file_ed25519(f, args.key)
            if sigp:
                print(str(sigp))
    return 0


def cmd_receipt(args: argparse.Namespace) -> int:
    meta = _read_json(args.metadata) if args.metadata else None
    out = generate_reproducibility_receipt(args.out, args.artifacts, extra_metadata=meta)
    print(str(out))
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    meta = _read_json(args.metadata)
    nodes, edges = lineage_from_training_metadata(meta)
    out = write_lineage_graph(args.out, nodes, edges, render_image=not args.no_image)
    print(str(out))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    base = Path(args.artifact).parent
    expect = [
        base / "MODEL_CARD.md",
        base / "MODEL_CARD.json",
        base / "SBOM.spdx.json",
        base / "CHECKSUMS.sha256",
        base / "RECEIPT.json",
        base / "LINEAGE.dot",
    ]
    missing = [p for p in expect if not p.exists()]
    if missing:
        for m in missing:
            print(f"MISSING: {m}", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="governance", description="Governance utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("card", help="Write model card")
    sp.add_argument("artifact", help="Path to primary artifact (model file or dir)")
    sp.add_argument("--metadata", required=True, help="Path to metadata JSON")
    sp.add_argument("--sbom", nargs="*", help="Optional SBOM paths to reference")
    sp.add_argument("--out", help="Output path for MODEL_CARD.md")
    sp.set_defaults(func=cmd_card)

    sp = sub.add_parser("sbom", help="Create SPDX SBOM from runtime packages")
    sp.add_argument("--out", required=True, help="Output path for SBOM JSON (e.g., SBOM.spdx.json)")
    sp.add_argument("--name", help="Base name for document")
    sp.set_defaults(func=cmd_sbom)

    sp = sub.add_parser("sign", help="Write checksums and optional signatures")
    sp.add_argument("--files", nargs="+", required=True, help="Files to checksum/sign")
    sp.add_argument("--out", required=True, help="Output path for CHECKSUMS.sha256")
    sp.add_argument("--key", help="Path to 32-byte Ed25519 private key (raw or base64)")
    sp.set_defaults(func=cmd_sign)

    sp = sub.add_parser("receipt", help="Generate reproducibility receipt")
    sp.add_argument("--artifacts", nargs="+", required=True, help="Artifacts to include")
    sp.add_argument("--out", required=True, help="Output path for RECEIPT.json")
    sp.add_argument("--metadata", help="Path to extra metadata JSON")
    sp.set_defaults(func=cmd_receipt)

    sp = sub.add_parser("lineage", help="Write lineage graph from training metadata")
    sp.add_argument("--metadata", required=True, help="Path to training metadata JSON")
    sp.add_argument("--out", required=True, help="Output .dot path")
    sp.add_argument("--no-image", action="store_true", help="Skip image rendering")
    sp.set_defaults(func=cmd_lineage)

    sp = sub.add_parser("verify", help="Verify required governance artifacts exist next to artifact")
    sp.add_argument("artifact", help="Primary artifact")
    sp.set_defaults(func=cmd_verify)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())


