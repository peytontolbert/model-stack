import torch

from runtime.seq2seq import EncoderDecoderLM
from specs.config import ModelConfig


def test_encoder_decoder_retrieval_heads_emit_normalized_vectors():
    cfg = ModelConfig(
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        vocab_size=128,
        norm="layer",
        max_position_embeddings=64,
        retrieval_head_dim=12,
    )
    model = EncoderDecoderLM(cfg, tie_embeddings=True, vocab_size=128)
    ids = torch.randint(1, 100, (2, 16))
    mask = torch.ones_like(ids)

    query = model.retrieval_query_embedding(ids, mask)
    doc = model.retrieval_doc_embedding(ids, mask)

    assert query.shape == (2, 12)
    assert doc.shape == (2, 12)
    assert torch.allclose(query.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(doc.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_encoder_decoder_agent_policy_heads_emit_paper_controller_logits():
    cfg = ModelConfig(
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        vocab_size=128,
        norm="layer",
        max_position_embeddings=64,
        agent_policy_heads=True,
    )
    model = EncoderDecoderLM(cfg, tie_embeddings=True, vocab_size=128)
    ids = torch.randint(1, 100, (2, 16))
    mask = torch.ones_like(ids)

    logits = model.agent_policy_logits(ids, mask)

    assert set(logits) == {
        "query_confidence",
        "retrieval_coverage",
        "ood_query",
        "ood_evidence",
        "answer_confidence",
        "needs_verification",
        "paper_action_validity",
    }
    assert all(value.shape == (2,) for value in logits.values())
