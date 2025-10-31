from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import ensure_directory, write_json


def _purl_for(name: str, version: str) -> str:
    return f"pkg:pypi/{name}@{version}"


def _spdx_id(name: str, version: str) -> str:
    return f"SPDXRef-Package-{name.replace('.', '-').replace('_', '-')}-{version}"


def _doc_name(base: Optional[str]) -> str:
    if base:
        return f"SBOM-{Path(base).stem}"
    return "SBOM-runtime"


def create_spdx_sbom(output_path: str | os.PathLike[str], base_name: Optional[str] = None) -> Path:
    now = datetime.now(timezone.utc).isoformat()

    packages: List[Dict[str, Any]] = []
    for dist in importlib_metadata.distributions():
        name = dist.metadata.get("Name") or dist.metadata.get("Summary") or dist.metadata.get("name") or "unknown"
        version = dist.version or "0.0.0"
        license_id = dist.metadata.get("License", "NOASSERTION") or "NOASSERTION"
        homepage = dist.metadata.get("Home-page", "")
        spdx_id = _spdx_id(name, version)
        pkg = {
            "name": name,
            "SPDXID": spdx_id,
            "versionInfo": version,
            "downloadLocation": "NOASSERTION",
            "licenseConcluded": license_id,
            "licenseDeclared": license_id,
            "supplier": "Organization: UNKNOWN",
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": _purl_for(name, version),
                }
            ],
        }
        if homepage:
            pkg["homepage"] = homepage
        packages.append(pkg)

    doc: Dict[str, Any] = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": _doc_name(base_name),
        "documentNamespace": f"spdx:sbom:{hashlib.sha1(str(output_path).encode()).hexdigest()}",
        "creationInfo": {
            "created": now,
            "creators": ["Tool: transformer_10-governance"],
        },
        "packages": packages,
    }

    outp = Path(output_path)
    ensure_directory(outp)
    write_json(outp, doc)
    return outp


