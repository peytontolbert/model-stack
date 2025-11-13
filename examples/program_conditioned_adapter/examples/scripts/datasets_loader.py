from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


def _read_json(path: Path) -> Dict[str, Any]:
	try:
		with open(path, "r", encoding="utf-8") as fh:
			return json.loads(fh.read())
	except Exception:
		return {}


def _load_local_jsonl(fp: Path, text_key: str = "text", max_n: int | None = None) -> List[str]:
	texts: List[str] = []
	if not fp.exists():
		return texts
	with open(fp, "r", encoding="utf-8") as fh:
		for line in fh:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except Exception:
				continue
			txt = str(obj.get(text_key) or "").strip()
			if txt:
				texts.append(txt)
			if max_n is not None and len(texts) >= int(max_n):
				break
	return texts


def _load_hf(repo: str, subset: str | None, split: str, max_n: int | None, cache_dir: Path) -> List[str]:
	try:
		from datasets import load_dataset  # type: ignore
	except Exception:
		return []
	ds = None
	try:
		if subset:
			ds = load_dataset(repo, subset, split=split, cache_dir=str(cache_dir))
		else:
			ds = load_dataset(repo, split=split, cache_dir=str(cache_dir))
	except Exception:
		return []
	texts: List[str] = []
	total = len(ds)
	for i in range(total):
		row = ds[i]
		prompt = str(row.get("prompt") or row.get("question") or row.get("text") or "").strip()
		code = str(row.get("code") or row.get("solution") or "").strip()
		combined = (prompt + ("\n" + code if code else "")).strip()
		if combined:
			texts.append(combined)
		if max_n is not None and len(texts) >= int(max_n):
			break
	return texts


def load_program_texts(example_dir: str, config_path: str | None = None, train_new_only: bool = False, state_path: str | None = None) -> Tuple[List[str], List[str]]:
	"""
	Load training texts from a config that can reference local and HF datasets.
	Priority: local first; if missing, fetch from HF and store under local_dir.

	Returns (texts, loaded_source_names)
	"""
	ex_dir = Path(example_dir).resolve()
	# Config discovery
	cfg_path = Path(config_path) if config_path else (ex_dir / "datasets" / "config.json")
	cfg = _read_json(cfg_path) if cfg_path.exists() else {}
	local_dir = Path(cfg.get("local_dir") or (ex_dir / "datasets")).resolve()
	local_dir.mkdir(parents=True, exist_ok=True)
	cache_dir = local_dir / "hf_cache"
	cache_dir.mkdir(parents=True, exist_ok=True)
	sources: List[Dict[str, Any]] = list(cfg.get("sources") or [])

	# Optional program state to support "train new only"
	seen: List[str] = []
	if train_new_only and state_path:
		try:
			st = _read_json(Path(state_path))
			seen = list(st.get("datasets_seen") or [])
		except Exception:
			seen = []

	texts_acc: List[str] = []
	loaded_names: List[str] = []
	for src in sources:
		name = str(src.get("name") or "").strip()
		if not name:
			continue
		if train_new_only and name in seen:
			continue
		max_n = src.get("max_n")
		text_key = str(src.get("text_key") or "text").strip()
		subset = src.get("subset")
		split = str(src.get("split") or "train").strip()
		# Determine local file path: explicit 'path' or derived from name+split
		rel = str(src.get("path") or "").strip()
		if rel:
			fp = Path(rel)
			if not fp.is_absolute():
				fp = (local_dir / rel).resolve()
		else:
			suffix = f"_{split}" if split else ""
			fp = (local_dir / f"{name.replace('/', '_')}{suffix}.jsonl").resolve()
		# Try local first
		out_texts: List[str] = _load_local_jsonl(fp, text_key=text_key, max_n=max_n)
		# If not present, try HF using 'name' as repo id
		if not out_texts:
			out_texts = _load_hf(name, subset, split, max_n, cache_dir)
			# Persist to local for future runs
			if out_texts:
				try:
					with open(fp, "w", encoding="utf-8") as fh:
						for t in out_texts:
							fh.write(json.dumps({"text": t}) + "\n")
				except Exception:
					pass
		if out_texts:
			texts_acc.extend(out_texts)
			loaded_names.append(f"{name}:{split}" if split else name)
	return texts_acc, loaded_names


