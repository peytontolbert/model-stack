from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from .utils import ensure_directory, try_import


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums_file(paths: Iterable[str | os.PathLike[str]], output_path: str | os.PathLike[str]) -> Path:
    outp = Path(output_path)
    ensure_directory(outp)
    with open(outp, "w", encoding="utf-8") as f:
        for p in paths:
            digest = sha256_file(p)
            f.write(f"{digest}  {Path(p)}\n")
    return outp


def sign_file_ed25519(
    file_path: str | os.PathLike[str],
    private_key_path: Optional[str | os.PathLike[str]],
    output_sig_path: Optional[str | os.PathLike[str]] = None,
) -> Optional[Path]:
    if private_key_path is None:
        return None
    nacl = try_import("nacl.signing")
    if nacl is None:
        return None
    from nacl.signing import SigningKey  # type: ignore

    with open(private_key_path, "rb") as f:
        key_bytes = f.read()
    # Expect raw 32-byte private key or a base64-encoded string
    if len(key_bytes) != 32:
        try:
            key_bytes = base64.b64decode(key_bytes)
        except Exception:
            return None
    if len(key_bytes) != 32:
        return None

    signing_key = SigningKey(key_bytes)
    data = Path(file_path).read_bytes()
    sig = signing_key.sign(data).signature
    sig_b64 = base64.b64encode(sig).decode("ascii")

    if output_sig_path is None:
        output_sig_path = str(Path(file_path).with_suffix(Path(file_path).suffix + ".sig"))
    outp = Path(output_sig_path)
    ensure_directory(outp)
    with open(outp, "w", encoding="utf-8") as f:
        f.write(sig_b64)
    return outp


