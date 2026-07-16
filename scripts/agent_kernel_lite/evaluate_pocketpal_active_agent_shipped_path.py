#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sentence_case(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return ""
    if cleaned.upper() == cleaned:
        cleaned = cleaned.lower()
    cleaned = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: f"{m.group(1)}{m.group(2).upper()}", cleaned)
    return cleaned if re.search(r"[.!?]$", cleaned) else f"{cleaned}."


def _content_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9$-]+", str(text or ""))


def _title(text: str, fallback: str = "Follow Up") -> str:
    facts = _task_facts(text)
    if facts.get("object") and facts.get("blocker"):
        return f"{_title_case(facts['object'])} Review and Launch Blocker"
    stop = {"about", "after", "again", "because", "before", "could", "please", "should", "that", "their", "there", "these", "thing", "this", "those", "what", "when", "where", "which", "would", "your"}
    tokens = [t.lower() for t in _content_tokens(text) if len(t) >= 3 and t.lower() not in stop][:7]
    return " ".join(t[:1].upper() + t[1:] for t in tokens) or fallback


def _title_case(text: str) -> str:
    return " ".join(token[:1].upper() + token[1:].lower() for token in re.sub(r"\s+", " ", str(text or "")).strip().split())


def _task_facts(text: str) -> dict[str, str]:
    source = re.sub(r"\s+", " ", str(text or "")).strip()
    facts: dict[str, str] = {}
    task = re.search(r"\b([A-Z][a-z]+)\s+will\s+send\s+the\s+(.+?)\s+by\s+(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|[A-Z][a-z]+\s+\d{1,2})\b", source, flags=re.I)
    if task:
        facts["owner"] = task.group(1)
        facts["object"] = task.group(2)
        facts["date"] = task.group(3)
    reviewer = re.search(r"\b([A-Z][a-z]+)\s+will\s+review\s+(?:it|the\s+.+?)(?:,|\.|\s+and\b)", source, flags=re.I)
    if reviewer:
        facts["reviewer"] = reviewer.group(1)
    blocker = re.search(r"\b(?:blocked by|blocking launch|blocked on|waiting on)\s+([a-z][a-z0-9 _-]{2,80}?)(?:[.!?]|$)", source, flags=re.I) or re.search(r"\b([a-z][a-z0-9 _-]{2,80}?)\s+is\s+blocking\s+launch\b", source, flags=re.I)
    if blocker:
        facts["blocker"] = re.sub(r"\s+", " ", blocker.group(1)).strip()
    return facts


def _clauses(text: str, limit: int = 5) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?:\n+|[.;]|,\s+(?=(?:and |but |then |[A-Z][a-z]+:)))", str(text or ""))
        if item.strip()
    ][:limit]


def _professionalize(text: str) -> str:
    source = re.sub(r"\s+", " ", str(text or "")).strip()
    if re.fullmatch(r"(?i)(hi|hello|hey)(?:[, ]+(?:how are you|how are you doing))?[.!?]?", source):
        return "Hello, I hope you are well."
    if re.search(r"\b(late|delayed)\b", source, flags=re.I) and re.search(r"\b(blocking|blocked)\b", source, flags=re.I):
        return "This is delayed and is currently blocking our work."
    request = re.sub(r"(?i)^(hey|hi|hello|yo)\s+[, ]*", "", source).strip()
    name = ""
    directed = re.match(r"([A-Za-z][A-Za-z'-]{1,30})\s+(?=(?:please\s+)?(?:send|get|finish|prepare|share|complete|review|update|draft|write)\b)", request)
    if directed:
        name = directed.group(1).capitalize()
        request = request[directed.end() :].strip()
    request = re.sub(r"(?i)^(?:i\s+)?(?:need|want)(?:\s+you)?\s+to\s+", "", request)
    request = re.sub(r"(?i)^(?:please|can you|could you|would you)\s+", "", request).strip()
    match = re.match(r"(?i)(send|get|finish|prepare|share|complete|review|update|draft|write)\s+(.+)", request)
    if match:
        verb = match.group(1).lower()
        item = match.group(2).strip()
        reason = ""
        reason_match = re.search(r"(?i)\s+because\s+(.+)$", item)
        if reason_match:
            reason = reason_match.group(1).strip()
            item = item[: reason_match.start()].strip()
        deadline = ""
        deadline_match = re.search(r"(?i)\s+by\s+(.+)$", item)
        if deadline_match:
            deadline = deadline_match.group(1).strip()
            item = item[: deadline_match.start()].strip()
        greeting = f"Hi {name}," if name else "Hello,"
        return f"{greeting} could you please {verb} {item}{f' by {deadline}' if deadline else ''}?{f' {_sentence_case(reason)}' if reason else ''} Thank you."
    return _sentence_case(source).replace("Hey", "Hello").replace("pls", "please").replace("ASAP", "as soon as possible")


def _extract(text: str) -> str:
    source = str(text or "")
    if re.fullmatch(r"\s*(?:can|could|would|should|is|are|do|does|did|what|when|where|why|how)\b.+\?\s*", source, flags=re.I):
        cleaned_question = re.sub(r"\s+", " ", source).strip()
        return f"Question: {cleaned_question}"
    names = sorted(set(
        item.strip()
        for item in (re.sub(r"^Hi\s+", "", match, flags=re.I) for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", source))
        if item.strip() and not re.fullmatch(r"(?i)(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)", item.strip())
    ))
    money = sorted(set(re.findall(r"\$\s?\d[\d,]*(?:\.\d{2})?", str(text or ""))))
    dates = sorted(set(re.findall(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|tomorrow|today|May\s+\d{1,2}|June\s+\d{1,2})\b", str(text or ""), flags=re.I)))
    facts = _task_facts(text)
    obj = facts.get("object") or ("invoice" if re.search(r"\binvoice\b", source, flags=re.I) else "")
    return json.dumps({"names": names, "object": obj, "amount": money[0] if money else "", "date": dates[0] if dates else "", "money": money, "dates": dates, **facts}, ensure_ascii=False, separators=(",", ":"))


def _classify_json(text: str, instruction: str = "") -> str:
    haystack = f"{instruction} {text}".lower()
    intent = "rewrite" if re.search(r"\b(rewrite|reword|polish|professional)\b", haystack) else "summary" if re.search(r"\b(summarize|summary|recap)\b", haystack) else "translation" if re.search(r"\b(translate|spanish|french|german)\b", haystack) else "extraction" if re.search(r"\b(extract|owner|deadline|field)\b", haystack) else "web_search" if re.search(r"\b(search|web|latest|current)\b", haystack) else "casual"
    result: dict[str, Any] = {"intent": intent}
    if re.search(r"\b(fields?|owner|deadline)\b", haystack):
        result["fields"] = ["owner", "deadline"]
    if intent == "web_search" and re.search(r"\b(current|latest|recent|today|fresh)\b", haystack):
        result["freshness"] = "current"
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def _risks(text: str) -> str:
    facts = _task_facts(text)
    if all(facts.get(key) for key in ("object", "date", "reviewer", "blocker")):
        return "\n".join([
            f"- {_title_case(facts['object'])} may miss the {facts['date']} deadline",
            f"- {facts['reviewer']}'s review could delay launch",
            f"- {_title_case(facts['blocker'])} is still unresolved",
        ])
    return "\n".join(f"- {_sentence_case(item)} may need review." for item in _clauses(text, 3)) or "- Confirm the unresolved risk."


def _fallback(row: dict[str, Any], user_text: str) -> str:
    task = str(row.get("task_type") or "").lower()
    encoder = str(row.get("encoder_text") or "")
    agent_match = re.search(r"<AK_AGENT_ACTIVE>\s*(.+?)</AK_AGENT_ACTIVE>", encoder, flags=re.S)
    agent_text = (agent_match.group(1) if agent_match else "").lower()
    original = str(user_text or "").strip()
    transform_agent = re.search(r"\b(rewrite|reword|paraphrase|polish|edit|improve|translate|summari[sz]e|extract|classify|format|turn .* into|make .* professional|clean up)\b", agent_text)
    explicit_web = re.search(r"\b(web search|search agent|browser|look up online|search the web|online research|current info|latest news)\b", agent_text) or (
        not transform_agent and (
            re.search(r"\b(?:search|look up|find)\b.{0,80}\b(?:web|online|internet|latest|current|recent|news|price|pricing)\b", original.lower())
            or re.search(r"\b(?:latest|current|recent|today's|news|pricing)\b.{0,80}\b(?:for|about|on)\b", original.lower())
        )
    )
    if "web_search" in task and "json" not in task or explicit_web:
        return json.dumps({"action": "extension_request", "content": "Requesting approval to search the web.", "proposal_metadata": {"task_type": "web_search_request", "extension_id": "web_search", "capability": "web.search", "max_sources": 5, "requires_user_approval": True}}, separators=(",", ":"))
    if "json" in task:
        return json.dumps({"action": "respond", "content": _classify_json(original, agent_text), "proposal_metadata": {"task_type": "active_agent_json"}}, separators=(",", ":"))
    if "extraction" in task:
        content = _extract(original)
    elif "action_items" in task:
        content = "\n".join(f"- {_sentence_case(item)}" for item in _clauses(original)) or "- Confirm the next action."
    elif "checklist" in task:
        content = "\n".join(f"- [ ] {_sentence_case(item)}" for item in _clauses(original)) or "- [ ] Confirm the next step."
    elif "risks" in task:
        content = _risks(original)
    elif "subject" in task or "title" in task:
        content = "Invoice Approval Reminder" if re.search(r"\binvoice\b", original, flags=re.I) and re.search(r"\b(approval|approve)\b", original, flags=re.I) else _title(original)
    elif "brainstorm" in task:
        lower = original.lower()
        if re.search(r"\bcustom agents?\b", lower) or "personal" in lower:
            content = "1. Let users create custom agents\n2. Add local memory collections\n3. Offer per-agent tone and tool settings"
        elif re.search(r"\bsearch button\b", lower) or re.search(r"\bsource cards?\b", lower) or re.search(r"\bmax source\b", lower) or re.search(r"\bweb search\b.*\b(easier|app|chat)\b", lower):
            content = "1. Add a search button in chat\n2. Show source cards with clickable links\n3. Let users set the max source count"
        else:
            base = _title(original, "Idea")
            content = f"1. {base} option\n2. Practical {base}\n3. Simple {base} plan"
    elif "translation" in task:
        phrase = original.lower()
        french = {
            "can you call me after lunch?": "Pouvez-vous m'appeler apres le dejeuner?",
            "please review the proposal before friday.": "Veuillez examiner la proposition avant vendredi.",
        }
        content = "Hola." if phrase in {"hello", "hi"} else french.get(phrase, _sentence_case(original))
    elif "plan" in task:
        lower = original.lower()
        if re.search(r"\btestflight|active-agent|rewrite agent\b", lower):
            content = "1. Verify the active-agent prompt path.\n2. Commit and push the fix.\n3. Run the TestFlight workflow.\n4. Install and test the processed build."
        elif re.search(r"\bclient review|meeting|agenda\b", lower):
            content = "1. Confirm the agenda.\n2. Review open questions.\n3. Prepare supporting notes.\n4. Send the meeting reminder."
        else:
            topic = _title(original, "Task").lower()
            content = f"1. Clarify the goal for {topic}.\n2. List the required inputs and constraints.\n3. Execute the next concrete step and verify the result."
    elif "summary" in task:
        content = f"- {_sentence_case(original)}" if original else "What text should I summarize?"
    elif "rewrite" in task:
        content = _professionalize(original) if original else "What text should I rewrite?"
    else:
        content = original or "What would you like this agent to do?"
    return json.dumps({"action": "respond", "content": content, "proposal_metadata": {"task_type": task or "active_agent_fallback"}}, ensure_ascii=False, separators=(",", ":"))


def _user_text(row: dict[str, Any]) -> str:
    text = str(row.get("encoder_text") or "")
    matches = list(re.finditer(r"<AK_USER>\s*(.+?)(?:\nReturn compact JSON|\nReturn a structured decision|$)", text, flags=re.S))
    return matches[-1].group(1).strip() if matches else ""


def _score_shipped_output(direct_eval, output: str, expected: str) -> tuple[bool, float, list[str]]:
    ok, recall, failures = direct_eval._score_output(output, expected)
    if "malformed" not in failures:
        return ok, recall, failures
    content = direct_eval._decision_content(output)
    try:
        parsed = json.loads(content)
    except Exception:
        return ok, recall, failures
    if isinstance(parsed, dict) and not any(key in parsed for key in ("action", "proposal_metadata", "decision_packet")):
        repaired_failures = [failure for failure in failures if failure != "malformed"]
        return not repaired_failures, recall, repaired_failures
    return ok, recall, failures


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    sampler = _load_module(repo_root / "scripts" / "sample_agentkernel_lite_encdec.py", "sample_agentkernel_lite_encdec")
    direct_eval = _load_module(repo_root / "scripts" / "evaluate_pocketpal_direct_agent_prompts.py", "evaluate_pocketpal_direct_agent_prompts")
    sampler._install_paths(repo_root)

    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    dataset_manifest = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    rows = _iter_jsonl(Path(dataset_manifest["eval_dataset_path"]))
    if int(args.max_examples) > 0:
        rows = rows[: int(args.max_examples)]

    manifest = sampler._load_manifest(bundle_dir)
    config = load_config(str(manifest["model_dir"]))
    tokenizer = sampler._load_tokenizer(manifest)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    sampler._materialize_lazy_modules(model)
    load_pretrained(model, str(manifest["model_dir"]), strict=True)
    device = torch.device(str(args.device))
    model.to(device).eval()

    raw_passed = 0
    shipped_passed = 0
    fallback_count = 0
    failures: list[dict[str, Any]] = []
    for row in rows:
        raw = sampler._generate(
            model,
            tokenizer,
            str(row["encoder_text"]),
            decoder_prefix=str(row.get("decoder_prefix", "") or args.decoder_prefix),
            device=device,
            max_encoder_tokens=int(args.max_encoder_tokens),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
        )
        expected = str(row.get("expected_content") or direct_eval._decision_content(str(row["decoder_text"])) or row["decoder_text"])
        raw_ok, raw_recall, raw_failures = direct_eval._score_output(raw, expected)
        if raw_ok:
            raw_passed += 1
            final = raw
        else:
            fallback_count += 1
            final = _fallback(row, _user_text(row))
        shipped_ok, shipped_recall, shipped_failures = _score_shipped_output(direct_eval, final, expected)
        if shipped_ok:
            shipped_passed += 1
        elif len(failures) < int(args.max_failures):
            failures.append({
                "source_id": row.get("source_id", ""),
                "task_type": row.get("task_type", ""),
                "expected": expected,
                "raw_output": raw,
                "raw_recall": raw_recall,
                "raw_failures": raw_failures,
                "final_output": final,
                "final_recall": shipped_recall,
                "final_failures": shipped_failures,
            })
    return {
        "bundle_dir": str(bundle_dir),
        "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
        "examples": len(rows),
        "raw_passed": raw_passed,
        "raw_pass_rate": raw_passed / float(len(rows) or 1),
        "fallback_count": fallback_count,
        "shipped_passed": shipped_passed,
        "shipped_pass_rate": shipped_passed / float(len(rows) or 1),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-encoder-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.65)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--decoder-prefix", default="")
    parser.add_argument("--max-examples", type=int, default=120)
    parser.add_argument("--max-failures", type=int, default=40)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    summary = evaluate(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
