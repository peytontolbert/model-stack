from __future__ import annotations

import os
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .runtime import ModelRuntime


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
    attention_mask: Optional[List[List[int]]] = Field(default=None, description="Optional token mask: shape [B, T]")
    max_new_tokens: int = 64
    do_sample: Optional[bool] = None
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    eos_id: Optional[int] = None
    no_repeat_ngram: int = 0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    sliding_window: Optional[int] = None
    beam_size: int = 1
    length_penalty: float = 1.0
    cache_backend: Optional[str] = None


class GenerateResponse(BaseModel):
    output_ids: List[List[int]]


app = FastAPI(title="Transformer Serve API")


@app.get("/healthz")
def healthz() -> dict:
    try:
        return get_runtime().health_info()
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    rt = get_runtime()
    try:
        cfg = rt.build_generation_config(
            max_new_tokens=req.max_new_tokens,
            do_sample=req.do_sample,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            eos_id=req.eos_id,
            no_repeat_ngram=req.no_repeat_ngram,
            repetition_penalty=req.repetition_penalty,
            presence_penalty=req.presence_penalty,
            frequency_penalty=req.frequency_penalty,
            sliding_window=req.sliding_window,
            beam_size=req.beam_size,
            length_penalty=req.length_penalty,
        )
        output_ids = rt.generate_token_lists(
            req.input_ids,
            config=cfg,
            attention_mask=req.attention_mask,
            cache_backend=req.cache_backend,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return GenerateResponse(output_ids=output_ids)


# Optional entrypoint for `uvicorn serve.api:app --host 0.0.0.0 --port 8000`
