## provenance.py — Git Provenance Helpers

Fetches commit and tree SHAs to include in manifests and optional answer footers.

### Key APIs
- `git_commit_sha(repo_root)` -> `Optional[str]`
- `git_tree_sha(repo_root)` -> `Optional[str]`

### Usage
Used by builders/runners to record provenance and by the runner to append an optional commit footer.


