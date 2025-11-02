import os
import sys
import subprocess
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def smoke_repo_path() -> Path:
    return repo_root() / "examples" / "repo_grounded_adapters" / "smoke_repo"


def _have_llama_snapshot(cache_dir: Path, model_id: str) -> bool:
    try:
        parts = model_id.strip().split("/")
        if len(parts) != 2:
            return False
        org, name = parts
        base = cache_dir / f"models--{org}--{name}" / "snapshots"
        if not base.is_dir():
            return False
        for snap in base.iterdir():
            if (snap / "config.json").is_file() and (
                (snap / "model.safetensors.index.json").is_file() or any(p.suffix == ".safetensors" for p in snap.glob("*.safetensors"))
            ):
                return True
        return False
    except Exception:
        return False


def run_smoke(prompt: str = "Explain how scaled_dot_product_attention works. Cite path:line.") -> int:
    """Run the enhanced runner on CPU against the tiny smoke repo.

    Returns the runner's exit code. Prints stdout/stderr for inspection.
    """
    repo = str(smoke_repo_path())
    # Prefer a local snapshot dir if provided; else use HF repo id
    model_arg = os.environ.get("LLAMA8B_LOCAL_SNAPSHOT", "meta-llama/Llama-3.1-8B-Instruct")
    if os.path.isdir(model_arg) and os.path.isfile(os.path.join(model_arg, "config.json")):
        print(f"[smoke] using local snapshot: {model_arg}")
    cache_dir = Path("/data/transformer_10/checkpoints")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Preflight: ensure weights are available when using the HF id
    if not (os.path.isdir(model_arg) and os.path.isfile(os.path.join(model_arg, "config.json"))):
        if not _have_llama_snapshot(cache_dir, model_arg):
            print("[smoke][error] LLaMA 8B weights not found in cache and no LLAMA8B_LOCAL_SNAPSHOT provided.")
            print("[smoke][action] Either set LLAMA8B_LOCAL_SNAPSHOT=/abs/path/to/local/snapshot (contains config.json and .safetensors) ")
            print("               or export HUGGINGFACE_HUB_TOKEN and re-run to download the weights into /data/transformer_10/checkpoints.")
            return 2
    # Baseline (no adapters)
    base_cmd = [
        sys.executable,
        "-m",
        "examples.repo_grounded_adapters.run",
        "--model", model_arg,
        "--repo", repo,
        "--prompt", prompt,
        "--pack-context",
        "--require-citations",
        "--device-map", "none",
        "--cache-dir", str(cache_dir),
        "--adapters-dir", str(repo_root() / "examples" / "repo_grounded_adapters" / "artifacts" / "base_adapters"),
        "--alpha", "0",
        "--rank", "8",
        "--gsub", "0.75",
        "--context-tokens", "800",
        "--min-new-tokens", "48",
        "--max-new-tokens", "160",
        "--seed", "0",
        "--no-adapters",
        "--verbose",
    ]
    # Adapted
    cmd = [
        sys.executable,
        "-m",
        "examples.repo_grounded_adapters.run",
        "--model", model_arg,
        "--repo", repo,
        "--prompt", prompt,
        "--pack-context",
        "--require-citations",
        "--device-map", "none",
        "--cache-dir", str(cache_dir),
        "--adapters-dir", str(repo_root() / "examples" / "repo_grounded_adapters" / "artifacts" / "base_adapters"),
        "--alpha", "10",
        "--rank", "8",
        "--gsub", "0.75",
        "--context-tokens", "800",
        "--min-new-tokens", "48",
        "--max-new-tokens", "160",
        "--seed", "0",
        "--target-weights", "q_proj=0,k_proj=0,v_proj=0",
        "--verbose",
    ]
    env = os.environ.copy()
    env["REPO_ADAPTER_DEBUG"] = "1"
    # Allow GPU usage by default; set SMOKE_FORCE_CPU=1 to force CPU
    if env.get("SMOKE_FORCE_CPU", "").strip():
        env["CUDA_VISIBLE_DEVICES"] = ""
    # Run baseline
    print("[smoke][baseline] running:", " ".join(base_cmd))
    proc_b = subprocess.run(base_cmd, env=env)
    # Run adapted
    print("[smoke][adapted] running:", " ".join(cmd))
    proc_a = subprocess.run(cmd, env=env)
    # Return adapted exit code if baseline succeeded, else baseline code
    return int(proc_a.returncode if proc_b.returncode == 0 else proc_b.returncode)


if __name__ == "__main__":
    code = run_smoke()
    sys.exit(code)


