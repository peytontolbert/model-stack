import torch

from runtime.asr.decode import AsrDecodeOptions, AsrDecodeState, apply_asr_logit_policy, greedy_decode_step


def test_apply_asr_logit_policy_suppresses_configured_tokens():
    logits = torch.tensor([[0.1, 0.2, 0.9, 0.4]])
    options = AsrDecodeOptions(suppress_token_ids=(2, 99))

    masked = apply_asr_logit_policy(logits, options)

    assert torch.isneginf(masked[0, 2])
    assert masked[0, 1] == logits[0, 1]


def test_greedy_decode_step_updates_completion_state():
    logits = torch.tensor([0.1, 0.4, 0.3])
    options = AsrDecodeOptions(eos_token_id=1, max_tokens=5)
    state = AsrDecodeState()

    token, state = greedy_decode_step(logits, state, options)

    assert token == 1
    assert state.tokens == [1]
    assert state.completed
