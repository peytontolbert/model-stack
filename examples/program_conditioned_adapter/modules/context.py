# function-first packer (lift from run_repo_adapter.py)

from typing import List, Optional, Tuple
import os
import torch  # type: ignore


def pack_context_heads(program_root: str, files: List[str], tok, budget_tokens: int) -> str:
    lines_out: List[str] = ["Program snippets:"]
    used = 0
    for rel in files:
        abs_fp = rel if os.path.isabs(rel) else os.path.abspath(os.path.join(program_root, rel))
        try:
            src_lines = open(abs_fp, "r", encoding="utf-8", errors="ignore").read().splitlines()
        except Exception:
            continue
        head_n = min(len(src_lines), 120)
        block = [f"[ctx] path: {os.path.relpath(abs_fp, program_root)}:1-{head_n}"] + src_lines[:head_n] + [""]
        text = "\n".join(block) + "\n"
        t = len(tok(text).input_ids)
        if used + t > budget_tokens:
            head_n = min(len(src_lines), 60)
            block = [f"[ctx] path: {os.path.relpath(abs_fp, program_root)}:1-{head_n}"] + src_lines[:head_n] + [""]
            text = "\n".join(block) + "\n"
            t = len(tok(text).input_ids)
            if used + t > budget_tokens:
                continue
        lines_out.extend(block)
        used += t
        if used >= budget_tokens:
            break
    return "\n".join(lines_out) if len(lines_out) > 1 else ""


def pack_context_windows(program_root: str, files: List[str], tok, budget_tokens: int) -> str:
    lines_out: List[str] = ["Program windows:"]
    used = 0
    for rel in files:
        abs_fp = rel if os.path.isabs(rel) else os.path.abspath(os.path.join(program_root, rel))
        try:
            src_lines = open(abs_fp, "r", encoding="utf-8", errors="ignore").read().splitlines()
        except Exception:
            continue
        def_lines: List[int] = []
        for i, line in enumerate(src_lines, start=1):
            s = line.lstrip()
            if s.startswith("def ") or s.startswith("class "):
                def_lines.append(i)
            if len(def_lines) >= 2:
                break
        def_lines = def_lines or [1]
        for ln in def_lines:
            a = max(1, ln - 20)
            b = min(len(src_lines), ln + 40)
            block = [f"[ctx] path: {os.path.relpath(abs_fp, program_root)}:{a}-{b}"] + src_lines[a - 1 : b] + [""]
            text = "\n".join(block) + "\n"
            t = len(tok(text).input_ids)
            if used + t > budget_tokens:
                continue
            lines_out.extend(block)
            used += t
            if used >= budget_tokens:
                break
        if used >= budget_tokens:
            break
    return "\n".join(lines_out) if len(lines_out) > 1 else ""


def collect_function_windows(program_root: str, files_: List[str], lines_each: int, *, max_candidates: int = 24) -> List[Tuple[str, int, int, int, List[str]]]:
    out: List[Tuple[str, int, int, int, List[str]]] = []
    for rel in files_:
        abs_fp = rel if os.path.isabs(rel) else os.path.abspath(os.path.join(program_root, rel))
        try:
            src_lines = open(abs_fp, "r", encoding="utf-8", errors="ignore").read().splitlines()
        except Exception:
            continue
        anchors: List[int] = []
        for i, line in enumerate(src_lines, start=1):
            s = line.lstrip()
            if s.startswith("def ") or s.startswith("class "):
                anchors.append(i)
                if len(anchors) >= max(4, int(max_candidates) // max(1, len(files_))):
                    break
        if not anchors:
            continue
        half = max(10, int(lines_each // 2))
        for ln in anchors:
            a = max(1, ln - half)
            b = min(len(src_lines), ln + half)
            out.append((rel, a, b, ln, src_lines[a - 1 : b]))
    return out


def extract_func_name_from_lines(lines_block: List[str], a: int, b: int, anchor_ln: int) -> Optional[str]:
    try:
        best_name = None
        best_dist = 10**9
        abs_ln = a
        import re as _re
        for ln_text in lines_block:
            s = ln_text.lstrip()
            if s.startswith("def ") or s.startswith("class "):
                m = _re.match(r"^(?:def|class)\s+([A-Za-z0-9_]+)", s)
                if m:
                    name = m.group(1)
                    dist = abs(abs_ln - anchor_ln)
                    if dist < best_dist:
                        best_dist = dist
                        best_name = name
            abs_ln += 1
        return best_name
    except Exception:
        return None



def score_yes_no(tok, model, q: str) -> float:
    q_ids = tok(q, return_tensors="pt")
    dev = next(model.parameters()).device
    q_ids = {k: v.to(dev) for k, v in q_ids.items()}
    with torch.no_grad():
        out_lm = model(**q_ids)
        # Support both HF-style outputs and raw tensor outputs
        if isinstance(out_lm, torch.Tensor):
            logits = out_lm
        else:
            logits = getattr(out_lm, "logits", None)
            if logits is None:
                # Best-effort fallbacks
                if isinstance(out_lm, (tuple, list)) and out_lm:
                    logits = out_lm[0]
                else:
                    logits = getattr(out_lm, "last_hidden_state", None)
            if logits is None:
                # As a last resort, return neutral 0.5
                return 0.5
        last = logits[:, -1, :]
        t1 = tok("1", add_special_tokens=False).input_ids
        t0 = tok("0", add_special_tokens=False).input_ids
        probs = torch.softmax(last, dim=-1)
        p1 = float(sum(probs[0, i].item() for i in (t1 or [])))
        p0 = float(sum(probs[0, i].item() for i in (t0 or [])))
        denom = max(1e-9, p1 + p0)
        return float(p1 / denom)


def model_prob_yes(tok, model, prompt_q: str, window_txt: str) -> Tuple[float, float]:
    rel_q = (
        "Question: " + prompt_q + "\nWindow:\n" + window_txt[:1800] + "\nDoes this window contain the core function or logic to answer the question? Answer 1 or 0."
    )
    noise_q = (
        "Question: " + prompt_q + "\nWindow:\n" + window_txt[:1800] + "\nIs this window likely test/tool/noise unrelated to answering the question? Answer 1 or 0."
    )
    return score_yes_no(tok, model, rel_q), score_yes_no(tok, model, noise_q)



""""""  # end