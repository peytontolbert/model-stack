import torch

from runtime.generation import greedy_generate as runtime_greedy_generate
from runtime.generation import sample_generate as runtime_sample_generate


@torch.no_grad()
def greedy_generate(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_id: int | None = None,
    attention_mask=None,
    attn_mask=None,
    no_repeat_ngram: int = 0,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    sliding_window: int | None = None,
    cache_backend: str | None = None,
) -> torch.Tensor:
    resolved_attention_mask = attention_mask if attention_mask is not None else attn_mask
    return runtime_greedy_generate(
        model,
        input_ids,
        max_new_tokens=int(max_new_tokens),
        eos_id=eos_id,
        attention_mask=resolved_attention_mask,
        no_repeat_ngram=int(no_repeat_ngram),
        repetition_penalty=float(repetition_penalty),
        presence_penalty=float(presence_penalty),
        frequency_penalty=float(frequency_penalty),
        sliding_window=(int(sliding_window) if sliding_window is not None else None),
        cache_backend=cache_backend,
    )


@torch.no_grad()
def sample_generate(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
    attention_mask=None,
    attn_mask=None,
    no_repeat_ngram: int = 0,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    sliding_window: int | None = None,
    cache_backend: str | None = None,
) -> torch.Tensor:
    resolved_attention_mask = attention_mask if attention_mask is not None else attn_mask
    return runtime_sample_generate(
        model,
        input_ids,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=top_k,
        top_p=top_p,
        eos_id=eos_id,
        attention_mask=resolved_attention_mask,
        no_repeat_ngram=int(no_repeat_ngram),
        repetition_penalty=float(repetition_penalty),
        presence_penalty=float(presence_penalty),
        frequency_penalty=float(frequency_penalty),
        sliding_window=(int(sliding_window) if sliding_window is not None else None),
        cache_backend=cache_backend,
    )
