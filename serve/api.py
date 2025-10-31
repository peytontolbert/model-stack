from __future__ import annotations

import os
from typing import Optional, List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .runtime import ModelRuntime
from .engine import generate as generate_tokens, GenerationConfig


_runtime: Optional[ModelRuntime] = None


def get_runtime() -> ModelRuntime:
    global _runtime
    if _runtime is None:
        model_dir = os.environ.get("MODEL_DIR")
        if not model_dir:
            raise RuntimeError("MODEL_DIR is not set; cannot initialize runtime")
        _runtime = ModelRuntime.from_dir(model_dir)
    return _runtime


class GenerateRequest(BaseModel):
    input_ids: List[List[int]] = Field(..., description="Batch of token id sequences: shape [B, T]")
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    eos_id: Optional[int] = None
    no_repeat_ngram: int = 0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class GenerateResponse(BaseModel):
    output_ids: List[List[int]]


app = FastAPI(title="Transformer Serve API")


@app.get("/healthz")
def healthz() -> dict:
    try:
        rt = get_runtime()
        return {"status": "ok", "device": str(rt.device), "dtype": str(rt.dtype)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    rt = get_runtime()
    try:
        ids = torch.tensor(req.input_ids, dtype=torch.long, device=rt.device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid input_ids: {e}")
    cache = rt.allocate_cache(batch_size=ids.shape[0])
    cfg = GenerationConfig(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        eos_id=req.eos_id,
        no_repeat_ngram=req.no_repeat_ngram,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
    )
    with torch.inference_mode():
        out = generate_tokens(rt.model, ids, cache=cache, config=cfg)
    return GenerateResponse(output_ids=out.tolist())


# Optional entrypoint for `uvicorn serve.api:app --host 0.0.0.0 --port 8000`


