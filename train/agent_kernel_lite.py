from __future__ import annotations

import argparse
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "agent_kernel_lite"


@dataclass(frozen=True)
class TrainingLane:
    name: str
    script: str
    description: str


LANES: tuple[TrainingLane, ...] = (
    TrainingLane(
        "diffusion-flow",
        "train_agentkernel_lite_image_flux_flow_distill.py",
        "Tiny FLUX packed-latent flow distillation student.",
    ),
    TrainingLane(
        "diffusion-rollout",
        "train_agentkernel_lite_image_flux_flow_rollout_distill.py",
        "Tiny FLUX rollout/on-policy distillation student.",
    ),
    TrainingLane(
        "diffusion-live-teacher",
        "train_agentkernel_lite_image_flux_live_teacher_flow.py",
        "Live-teacher FLUX distillation loop.",
    ),
    TrainingLane(
        "sana-latent",
        "train_agentkernel_lite_image_sana_latent_distill.py",
        "SANA latent distillation student.",
    ),
    TrainingLane(
        "f5tts-distill",
        "distill_f5tts_q4_teacher.py",
        "F5TTS Q4 teacher/student distillation.",
    ),
    TrainingLane(
        "f5tts-streaming",
        "train_f5tts_q4_streaming.py",
        "F5TTS Q4 streaming fine-tuning loop.",
    ),
    TrainingLane(
        "seq2seq",
        "train_agentkernel_lite_encdec.py",
        "Tiny encoder-decoder / PocketPal-style seq2seq training.",
    ),
)


def lane_map() -> dict[str, TrainingLane]:
    return {lane.name: lane for lane in LANES}


def resolve_script(name_or_path: str) -> Path:
    lanes = lane_map()
    if name_or_path in lanes:
        candidate = SCRIPT_ROOT / lanes[name_or_path].script
    else:
        raw = Path(name_or_path)
        candidate = raw if raw.is_absolute() else SCRIPT_ROOT / raw
    candidate = candidate.resolve()
    try:
        candidate.relative_to(SCRIPT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"script must live under {SCRIPT_ROOT}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def list_lanes() -> str:
    width = max(len(lane.name) for lane in LANES)
    rows = ["Available Agent Kernel Lite training lanes:"]
    for lane in LANES:
        rows.append(f"  {lane.name:<{width}}  {lane.script}  - {lane.description}")
    rows.append("")
    rows.append("Run an imported script by alias:")
    rows.append("  python -m train.agent_kernel_lite run seq2seq -- --help")
    rows.append("")
    rows.append("Run any imported script by file name:")
    rows.append("  python -m train.agent_kernel_lite run build_pocketpal_stage1_agent_dataset.py -- --help")
    return "\n".join(rows)


def run_script(script: Path, script_args: list[str]) -> None:
    script_root = str(SCRIPT_ROOT)
    repo_root = str(REPO_ROOT)
    for entry in (repo_root, script_root):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script), *script_args]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m train.agent_kernel_lite",
        description="Run imported Agent Kernel Lite training scripts through model-stack.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("list", help="List curated training lane aliases.")
    run = sub.add_parser("run", help="Run an imported script alias or file name.")
    run.add_argument("script", help="Curated lane alias or script file under scripts/agent_kernel_lite.")
    run.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the target script after --.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in (None, "list"):
        print(list_lanes())
        return 0
    if args.command == "run":
        script_args = list(args.script_args)
        if script_args[:1] == ["--"]:
            script_args = script_args[1:]
        script = resolve_script(args.script)
        run_script(script, script_args)
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
