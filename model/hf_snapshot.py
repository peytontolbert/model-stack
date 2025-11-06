
def ensure_snapshot(model_id: str, cache_dir: str) -> str:
    try:
        import os as _os
        if _os.path.isdir(model_id) and _os.path.isfile(_os.path.join(model_id, "config.json")):
            return model_id
        try:
            org_name = model_id.strip().split("/")[-2:]
            if len(org_name) == 2:
                org, name = org_name
                dir1 = _os.path.join(cache_dir, f"models--{org}--{name}", "snapshots")
                candidates: list[str] = []
                if _os.path.isdir(dir1):
                    candidates.extend([_os.path.join(dir1, d) for d in _os.listdir(dir1)])
                else:
                    root = _os.path.join(cache_dir)
                    for d in _os.listdir(root):
                        if not d.startswith("models--"):
                            continue
                        snap = _os.path.join(root, d, "snapshots")
                        if _os.path.isdir(snap):
                            for sd in _os.listdir(snap):
                                candidates.append(_os.path.join(snap, sd))
                candidates = [p for p in candidates if _os.path.isfile(_os.path.join(p, "config.json"))]
                if candidates:
                    candidates.sort(key=lambda p: _os.path.getmtime(p), reverse=True)
                    return candidates[0]
        except Exception:
            pass
        try:
            from huggingface_hub import snapshot_download as _snap
        except Exception:
            raise RuntimeError("huggingface_hub is required to download the model snapshot")
        return _snap(repo_id=model_id, cache_dir=cache_dir)
    except Exception as e:
        raise e
