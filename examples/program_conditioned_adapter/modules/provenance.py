import subprocess
from typing import Optional


def git_commit_sha(repo_root: str) -> Optional[str]:
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return sha if sha else None
    except Exception:
        return None

def git_tree_sha(repo_root: str) -> Optional[str]:
    """Return the HEAD tree SHA if available (pins exact tracked file set)."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD^{tree}"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return sha if sha else None
    except Exception:
        return None
