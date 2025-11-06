from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import json


class BaseTokenizer:
    def encode(self, text: str) -> List[int]:  # pragma: no cover - interface
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def info(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class HFTokenizer(BaseTokenizer):
    def __init__(self, name_or_path: str) -> None:
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            raise RuntimeError("Install 'transformers' to use HFTokenizer") from e
        self._name = str(name_or_path)
        self._tok = AutoTokenizer.from_pretrained(self._name)
        if getattr(self._tok, "pad_token", None) is None:
            try:
                self._tok.pad_token = self._tok.eos_token
            except Exception:
                pass

    def encode(self, text: str) -> List[int]:
        return list(self._tok.encode(text, add_special_tokens=False))

    def decode(self, ids: List[int]) -> str:
        return str(self._tok.decode(ids))

    def info(self) -> Dict[str, Any]:
        return {"type": "hf", "name_or_path": self._name}


class LocalLlamaTokenizer(BaseTokenizer):
    def __init__(self, snapshot_dir: str) -> None:
        self._root = os.path.abspath(snapshot_dir)
        if not os.path.isdir(self._root):
            raise FileNotFoundError(f"snapshot_dir not found: {self._root}")
        # Try tokenizers JSON first, then SentencePiece model
        self._tok_json = os.path.join(self._root, "tokenizer.json")
        self._sp_model = os.path.join(self._root, "tokenizer.model")
        self._backend = None
        self._tok = None
        if os.path.isfile(self._tok_json):
            try:
                from tokenizers import Tokenizer  # type: ignore
            except Exception as e:
                raise RuntimeError("Install 'tokenizers' to load tokenizer.json") from e
            self._tok = Tokenizer.from_file(self._tok_json)
            self._backend = "tokenizers"
        elif os.path.isfile(self._sp_model):
            try:
                import sentencepiece as spm  # type: ignore
            except Exception as e:
                raise RuntimeError("Install 'sentencepiece' to load tokenizer.model") from e
            sp = spm.SentencePieceProcessor(model_file=self._sp_model)
            self._tok = sp
            self._backend = "sentencepiece"
        else:
            raise FileNotFoundError(f"No tokenizer.json or tokenizer.model found under {self._root}")
        # Special tokens (best-effort from config files)
        self._bos_id: Optional[int] = None
        self._eos_id: Optional[int] = None
        self._unk_id: Optional[int] = None
        self._pad_id: Optional[int] = None
        self._special_token_strs: Dict[str, str] = {}
        self._load_special_tokens()

    def _load_special_tokens(self) -> None:
        # Read config.json, tokenizer_config.json and special_tokens_map.json if present
        cfg: Dict[str, Any] = {}
        model_cfg: Dict[str, Any] = {}
        spmap: Dict[str, Any] = {}
        try:
            path = os.path.join(self._root, "config.json")
            if os.path.isfile(path):
                model_cfg = json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            model_cfg = {}
        try:
            path = os.path.join(self._root, "tokenizer_config.json")
            if os.path.isfile(path):
                cfg = json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            cfg = {}
        try:
            path = os.path.join(self._root, "special_tokens_map.json")
            if os.path.isfile(path):
                spmap = json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            spmap = {}
        # Prefer explicit ids in config
        def _get_id(obj: Dict[str, Any], key: str) -> Optional[int]:
            v = obj.get(key)
            if isinstance(v, int):
                return int(v)
            try:
                if isinstance(v, dict) and "id" in v:
                    return int(v["id"])  # type: ignore
            except Exception:
                return None
            return None
        # From model config first
        self._bos_id = _get_id(model_cfg, "bos_token_id")
        self._eos_id = _get_id(model_cfg, "eos_token_id")
        self._unk_id = _get_id(model_cfg, "unk_token_id")
        self._pad_id = _get_id(model_cfg, "pad_token_id")
        # From tokenizer config if still missing
        self._bos_id = self._bos_id or _get_id(cfg, "bos_token_id")
        self._eos_id = self._eos_id or _get_id(cfg, "eos_token_id")
        self._unk_id = self._unk_id or _get_id(cfg, "unk_token_id")
        self._pad_id = self._pad_id or _get_id(cfg, "pad_token_id")
        # Collect special token strings
        def _get_token_str(obj: Dict[str, Any], key: str) -> Optional[str]:
            v = obj.get(key)
            if isinstance(v, str):
                return v
            if isinstance(v, dict):
                s = v.get("content") or v.get("name") or v.get("token")
                if isinstance(s, str):
                    return s
            return None
        for k in ("bos_token", "eos_token", "unk_token", "pad_token"):
            s = _get_token_str(cfg, k)
            if s is None:
                s = _get_token_str(spmap, k)
            if s is not None:
                self._special_token_strs[k] = s
        # Resolve missing ids by token strings via backend
        def _resolve_id_by_string(tok_str: str) -> Optional[int]:
            try:
                if self._backend == "tokenizers":
                    tid = self._tok.token_to_id(tok_str)  # type: ignore[attr-defined]
                    return int(tid) if tid is not None else None
                # sentencepiece
                tid = self._tok.PieceToId(tok_str)  # type: ignore[attr-defined]
                return int(tid) if isinstance(tid, int) and tid >= 0 else None
            except Exception:
                return None
        if self._bos_id is None and (t := self._special_token_strs.get("bos_token")):
            self._bos_id = _resolve_id_by_string(t)
        if self._eos_id is None and (t := self._special_token_strs.get("eos_token")):
            self._eos_id = _resolve_id_by_string(t)
        if self._unk_id is None and (t := self._special_token_strs.get("unk_token")):
            self._unk_id = _resolve_id_by_string(t)
        if self._pad_id is None and (t := self._special_token_strs.get("pad_token")):
            self._pad_id = _resolve_id_by_string(t)
        # Common LLaMA convention: pad_token_id equals eos_token_id when undefined
        if self._pad_id is None and self._eos_id is not None:
            self._pad_id = int(self._eos_id)

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._bos_id

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._eos_id

    @property
    def unk_token_id(self) -> Optional[int]:
        return self._unk_id

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._pad_id

    def encode(self, text: str) -> List[int]:
        if self._backend == "tokenizers":
            # add_special_tokens=False to mirror HFTokenizer.encode
            return list(self._tok.encode(text, add_special_tokens=False).ids)  # type: ignore[attr-defined]
        # sentencepiece path
        ids = self._tok.Encode(text, out_type=int)  # type: ignore[attr-defined]
        return list(ids)

    def decode(self, ids: List[int], *, skip_special_tokens: bool = False) -> str:
        if self._backend == "tokenizers":
            return str(self._tok.decode(ids, skip_special_tokens=bool(skip_special_tokens)))  # type: ignore[attr-defined]
        if not skip_special_tokens:
            return str(self._tok.DecodeIds(list(ids)))  # type: ignore[attr-defined]
        # sentencepiece path with special removal by string match (best-effort)
        try:
            toks = [self._tok.IdToPiece(int(i)) for i in ids]  # type: ignore[attr-defined]
            specials = set(self._special_token_strs.values())
            toks = [t for t in toks if t not in specials]
            # naive join; sentencepiece decoding with removed specials may differ slightly
            return "".join(toks)
        except Exception:
            return str(self._tok.DecodeIds(list(ids)))  # type: ignore[attr-defined]


class PureLlamaTokenizer(BaseTokenizer):
    """Pure-Python Unigram tokenizer for LLaMA snapshots using tokenizer.json only.

    - No external dependencies (no tokenizers/sentencepiece).
    - Implements decode exactly; implements encode via Viterbi over the Unigram vocab.
    - Handles basic whitespace as SentencePiece: inserts '▁' at word starts.
    """

    def __init__(self, snapshot_dir: str) -> None:
        self._root = os.path.abspath(snapshot_dir)
        path = os.path.join(self._root, "tokenizer.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"tokenizer.json not found under {self._root}")
        data = json.load(open(path, "r", encoding="utf-8"))
        model = data.get("model", {})
        self._model_type = str(model.get("type") or "").lower()
        self._id_to_piece: List[str] = []
        self._piece_to_id: Dict[str, int] = {}
        self._piece_score: Dict[str, float] = {}
        # Pre-tokenizer/normalizer hints
        self._byte_level: bool = False
        self._byte_add_prefix_space: bool = False
        self._metaspace_char: Optional[str] = None
        self._metaspace_add_prefix_space: bool = False
        self._unicode_norm: Optional[str] = None  # e.g., 'NFC' or 'NFKC'
        def _scan_pre_tok(pt):
            if pt is None:
                return
            if isinstance(pt, dict):
                t = str(pt.get("type") or pt.get("@type") or "")
                if t.endswith("ByteLevel"):
                    self._byte_level = True
                    self._byte_add_prefix_space = bool(pt.get("add_prefix_space", False))
                if t.endswith("Metaspace"):
                    self._metaspace_char = pt.get("replacement", self._metaspace_char)
                    self._metaspace_add_prefix_space = bool(pt.get("add_prefix_space", self._metaspace_add_prefix_space))
                # Sequence or Wrap
                for k in ("pretokenizers", "pre_tokenizers", "pre_tokenizer", "children"):
                    v = pt.get(k)
                    if isinstance(v, list):
                        for x in v:
                            _scan_pre_tok(x)
                    elif isinstance(v, dict):
                        _scan_pre_tok(v)
            elif isinstance(pt, list):
                for x in pt:
                    _scan_pre_tok(x)
        _scan_pre_tok(data.get("pre_tokenizer"))
        # Scan normalizer
        def _scan_norm(nm):
            if nm is None:
                return
            if isinstance(nm, dict):
                t = str(nm.get("type") or nm.get("@type") or "")
                if t.endswith("NFC"):
                    self._unicode_norm = "NFC"
                if t.endswith("NFKC"):
                    self._unicode_norm = "NFKC"
                for k in ("normalizers", "normalizer", "children"):
                    v = nm.get(k)
                    if isinstance(v, list):
                        for x in v:
                            _scan_norm(x)
                    elif isinstance(v, dict):
                        _scan_norm(v)
            elif isinstance(nm, list):
                for x in nm:
                    _scan_norm(x)
        _scan_norm(data.get("normalizer"))
        if self._model_type == "unigram":
            vocab_list = model.get("vocab", [])
            # vocab_list: list of [piece, score]
            for i, item in enumerate(vocab_list):
                if not isinstance(item, list) or len(item) < 1:
                    continue
                piece = str(item[0])
                score = float(item[1]) if len(item) > 1 else 0.0
                self._id_to_piece.append(piece)
                self._piece_to_id[piece] = i
                self._piece_score[piece] = score
        else:
            # BPE or other types: expect a vocab dict mapping token->id
            vocab_dict = model.get("vocab", {})
            if not isinstance(vocab_dict, dict) or len(vocab_dict) == 0:
                raise RuntimeError("Unsupported tokenizer.json format: missing vocab")
            # Ensure ordering by id
            items = sorted(vocab_dict.items(), key=lambda kv: int(kv[1]))
            for piece, idx in items:
                i = int(idx)
                # grow list to size
                if i >= len(self._id_to_piece):
                    self._id_to_piece.extend([""] * (i - len(self._id_to_piece) + 1))
                self._id_to_piece[i] = str(piece)
                self._piece_to_id[str(piece)] = i
            # No scores; leave empty to trigger greedy longest-match
        # Added tokens (specials) with fixed ids
        self._added_specials: Dict[str, int] = {}
        try:
            added = data.get("added_tokens", [])
            if isinstance(added, list):
                for at in added:
                    if not isinstance(at, dict):
                        continue
                    if not at.get("special", False):
                        continue
                    content = str(at.get("content", ""))
                    tid = at.get("id")
                    if isinstance(tid, int) and content:
                        i = int(tid)
                        if i >= len(self._id_to_piece):
                            self._id_to_piece.extend([""] * (i - len(self._id_to_piece) + 1))
                        self._id_to_piece[i] = content
                        self._piece_to_id[content] = i
                        self._added_specials[content] = i
        except Exception:
            pass
        spmap_path = os.path.join(self._root, "special_tokens_map.json")
        self._specials: Dict[str, int] = {}
        if os.path.isfile(spmap_path):
            try:
                spmap = json.load(open(spmap_path, "r", encoding="utf-8"))
                for k in ("bos_token", "eos_token", "unk_token", "pad_token"):
                    v = spmap.get(k)
                    if isinstance(v, str) and v in self._piece_to_id:
                        self._specials[k] = int(self._piece_to_id[v])
            except Exception:
                pass
        # Also set from config.json numeric ids if present (authoritative)
        try:
            cfg_path = os.path.join(self._root, "config.json")
            if os.path.isfile(cfg_path):
                cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
                def _vid_as_int(val):
                    if isinstance(val, int):
                        return int(val)
                    if isinstance(val, list) and val:
                        # HF sometimes stores [special_eos, generation_eos]; prefer last as in HF
                        last = val[-1]
                        if isinstance(last, int):
                            return int(last)
                    return None
                for key in ("bos_token_id", "eos_token_id", "unk_token_id", "pad_token_id"):
                    vid = _vid_as_int(cfg.get(key))
                    if isinstance(vid, int) and vid >= 0:
                        if key == "bos_token_id":
                            self._specials["bos_token"] = int(vid)
                        elif key == "eos_token_id":
                            self._specials["eos_token"] = int(vid)
                        elif key == "unk_token_id":
                            self._specials["unk_token"] = int(vid)
                        elif key == "pad_token_id":
                            self._specials["pad_token"] = int(vid)
        except Exception:
            pass
        # Also infer from common LLaMA special names
        for k, sym in ("bos_token", "<|begin_of_text|>"), ("eos_token", "<|end_of_text|>"), ("unk_token", "<unk>"), ("pad_token", "<pad>"):
            if k not in self._specials and sym in self._piece_to_id:
                self._specials[k] = int(self._piece_to_id[sym])
        # Build a trie over pieces for encoding (all vocab tokens)
        self._trie: Dict[str, Any] = {}
        for piece in self._piece_to_id.keys():
            node = self._trie
            for ch in piece:
                node = node.setdefault(ch, {})
            node["__id__"] = self._piece_to_id[piece]
            if self._model_type == "unigram":
                node["__score__"] = self._piece_score.get(piece, 0.0)
        # Byte-level reversible maps
        self._b2u: Dict[int, str] = {}
        self._u2b: Dict[str, int] = {}
        if self._byte_level:
            def bytes_to_unicode():
                bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
                cs = bs[:]
                n = 0
                for b in range(256):
                    if b not in bs:
                        bs.append(b)
                        cs.append(256 + n)
                        n += 1
                cs = [chr(n) for n in cs]
                return dict(zip(bs, cs))
            self._b2u = bytes_to_unicode()
            self._u2b = {v: k for k, v in self._b2u.items()}

    def _normalize(self, text: str) -> str:
        # Metaspace-style: replace every space with replacement marker; optionally prefix
        if text is None:
            return ""
        s = text
        # Unicode normalization first if configured
        try:
            if self._unicode_norm is not None:
                import unicodedata as _ud
                s = _ud.normalize(self._unicode_norm, s)
        except Exception:
            pass
        if self._metaspace_char is not None:
            rep = str(self._metaspace_char)
            # Prefix if requested and not starting with whitespace
            if self._metaspace_add_prefix_space and (len(s) == 0 or not s[0].isspace()):
                s = rep + s
            # Replace spaces with replacement; keep tabs/newlines intact
            s = s.replace(' ', rep)
            return s
        # Fallback: simple space-to-▁ mapping
        s = s.replace(' ', '▁')
        return s

    def decode(self, ids: List[int]) -> str:
        pieces = [self._id_to_piece[i] if 0 <= int(i) < len(self._id_to_piece) else "" for i in ids]
        s = "".join(pieces)
        if self._byte_level and self._u2b:
            b = bytearray()
            for ch in s:
                b.append(self._u2b.get(ch, ord('?')))
            try:
                return b.decode("utf-8", errors="replace")
            except Exception:
                return b.decode("latin-1", errors="replace")
        # Metaspace / Unigram path
        return s.replace('▁', ' ')

    def _split_by_special(self, text: str) -> List[tuple[bool, str | int]]:
        if not self._added_specials:
            return [(False, text)]
        # Greedy longest-match of special token contents
        specials = sorted(self._added_specials.keys(), key=len, reverse=True)
        out: List[tuple[bool, str | int]] = []
        i = 0
        n = len(text)
        while i < n:
            matched = False
            for s in specials:
                if text.startswith(s, i):
                    out.append((True, int(self._added_specials[s])))
                    i += len(s)
                    matched = True
                    break
            if not matched:
                # accumulate until next special
                j = i + 1
                while j <= n and all(not text.startswith(s, j) for s in specials):
                    j += 1
                out.append((False, text[i:j]))
                i = j
        return out

    def _normalize_stream(self, text: str) -> str:
        if self._byte_level and self._b2u:
            s = text
            if self._byte_add_prefix_space and (len(s) == 0 or s[0] != ' '):
                s = ' ' + s
            raw = s.encode("utf-8")
            return ''.join(self._b2u.get(b, '?') for b in raw)
        return self._normalize(text)

    def encode(self, text: str) -> List[int]:
        # Split around special tokens so they are preserved as single ids
        parts = self._split_by_special(text)
        ids_total: List[int] = []
        for is_special, payload in parts:
            if is_special:
                ids_total.append(int(payload))
                continue
            segment = str(payload)
            norm = self._normalize_stream(segment)
            # Fallback: empty segments contribute nothing
            if not norm:
                continue
            # existing Unigram/BPE encode on normalized stream
            # (rest of function continues)
            #
            # Apply current algorithm on 'norm' below
            
            # existing code moved below with 'norm' defined
            
            # Begin original per-stream encoding
            n = len(norm)
            if self._model_type == "unigram":
                INF = 1e30
                best_cost = [INF] * (n + 1)
                best_cost[0] = 0.0
                prev = [-1] * (n + 1)
                prev_len = [0] * (n + 1)
                for i in range(n):
                    if best_cost[i] >= INF:
                        continue
                    node = self._trie
                    j = i
                    while j < n and norm[j] in node:
                        node = node[norm[j]]
                        j += 1
                        if "__id__" in node:
                            pid = node["__id__"]
                            score = node.get("__score__", 0.0)
                            cost = best_cost[i] - float(score)
                            if cost < best_cost[j]:
                                best_cost[j] = cost
                                prev[j] = pid
                                prev_len[j] = j - i
                    if prev[i + 1] == -1 and i + 1 <= n:
                        unk_id = self._specials.get("unk_token", self._piece_to_id.get("<unk>", -1))
                        if unk_id >= 0:
                            cost = best_cost[i] + 10.0
                            if cost < best_cost[i + 1]:
                                best_cost[i + 1] = cost
                                prev[i + 1] = unk_id
                                prev_len[i + 1] = 1
                if best_cost[n] < INF:
                    pos = n
                    seg_ids: List[int] = []
                    while pos > 0:
                        pid = prev[pos]
                        l = prev_len[pos]
                        if pid == -1 or l <= 0:
                            break
                        seg_ids.append(pid)
                        pos -= l
                    seg_ids.reverse()
                    ids_total.extend(seg_ids)
                continue
            # BPE greedy longest-match
            i = 0
            while i < n:
                node = self._trie
                j = i
                last_id = -1
                last_j = i
                while j < n and norm[j] in node:
                    node = node[norm[j]]
                    j += 1
                    if "__id__" in node:
                        last_id = node["__id__"]
                        last_j = j
                if last_id >= 0:
                    ids_total.append(last_id)
                    i = last_j
                else:
                    unk_id = self._specials.get("unk_token", self._piece_to_id.get("<unk>", -1))
                    if unk_id >= 0:
                        ids_total.append(unk_id)
                    i += 1
        return ids_total
        n = len(norm)
        if self._model_type == "unigram":
            # Viterbi DP: best cost and backpointers
            INF = 1e30
            best_cost = [INF] * (n + 1)
            best_cost[0] = 0.0
            prev = [-1] * (n + 1)
            prev_len = [0] * (n + 1)
            # Traverse positions, extend by matches in trie
            for i in range(n):
                if best_cost[i] >= INF:
                    continue
                node = self._trie
                j = i
                while j < n and norm[j] in node:
                    node = node[norm[j]]
                    j += 1
                    if "__id__" in node:
                        pid = node["__id__"]
                        score = node.get("__score__", 0.0)
                        # Unigram uses negative log-likelihood-like scores; lower score is better
                        cost = best_cost[i] - float(score)
                        if cost < best_cost[j]:
                            best_cost[j] = cost
                            prev[j] = pid
                            prev_len[j] = j - i
                # Fallback to unknown for single char if no match advanced
                if prev[i + 1] == -1 and i + 1 <= n:
                    unk_id = self._specials.get("unk_token", self._piece_to_id.get("<unk>", -1))
                    if unk_id >= 0:
                        cost = best_cost[i] + 10.0  # penalty for unk
                        if cost < best_cost[i + 1]:
                            best_cost[i + 1] = cost
                            prev[i + 1] = unk_id
                            prev_len[i + 1] = 1
            # Reconstruct
            if best_cost[n] >= INF:
                return []
            ids: List[int] = []
            pos = n
            while pos > 0:
                pid = prev[pos]
                l = prev_len[pos]
                if pid == -1 or l <= 0:
                    break
                ids.append(pid)
                pos -= l
            ids.reverse()
            return ids
        # BPE/other: greedy longest-match using vocab trie
        ids: List[int] = []
        i = 0
        while i < n:
            node = self._trie
            j = i
            last_id = -1
            last_j = i
            while j < n and norm[j] in node:
                node = node[norm[j]]
                j += 1
                if "__id__" in node:
                    last_id = node["__id__"]
                    last_j = j
            if last_id >= 0:
                ids.append(last_id)
                i = last_j
            else:
                # unknown single character
                unk_id = self._specials.get("unk_token", self._piece_to_id.get("<unk>", -1))
                if unk_id >= 0:
                    ids.append(unk_id)
                i += 1
        return ids

    def info(self) -> Dict[str, Any]:
        return {
            "type": "pure_llama",
            "snapshot_dir": self._root,
            "bos": self.bos_token_id,
            "eos": self.eos_token_id,
            "unk": self.unk_token_id,
            "pad": self.pad_token_id,
        }

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._specials.get("bos_token")

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._specials.get("eos_token")

    @property
    def unk_token_id(self) -> Optional[int]:
        return self._specials.get("unk_token")

    @property
    def pad_token_id(self) -> Optional[int]:
        pid = self._specials.get("pad_token")
        if pid is None:
            pid = self._specials.get("eos_token")
        return pid


class WhitespaceTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        # Extremely simple and not suitable for production; stable mapping per run
        self._vocab: Dict[str, int] = {}
        self._unk = 0
        self._vocab["<unk>"] = self._unk

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for tok in text.strip().split():
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab)
            ids.append(self._vocab.get(tok, self._unk))
        return ids

    def decode(self, ids: List[int]) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        return " ".join(inv.get(i, "<unk>") for i in ids)

    def info(self) -> Dict[str, Any]:
        return {"type": "whitespace"}


def get_tokenizer(name_or_path: Optional[str] = None) -> BaseTokenizer:
    if name_or_path is None or str(name_or_path).strip() == "":
        return WhitespaceTokenizer()
    return HFTokenizer(str(name_or_path))


__all__ = ["BaseTokenizer", "HFTokenizer", "LocalLlamaTokenizer", "PureLlamaTokenizer", "WhitespaceTokenizer", "get_tokenizer"]


