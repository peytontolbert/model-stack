import torch
from metrics import masked_accuracy, masked_token_f1, ece_binning, sequence_logprob
from losses import masked_focal_loss, masked_bce_multilabel, masked_entropy_from_logits


def test_metrics_and_losses_shapes():
    logits = torch.randn(2, 4, 6)
    target = torch.randint(0, 6, (2, 4))
    mask = torch.tensor([[False, False, True, True], [False, True, True, True]])
    acc = masked_accuracy(logits, target, mask)
    f1 = masked_token_f1(logits.argmax(-1), target, mask)
    ece = ece_binning(logits, target, mask)
    lp = sequence_logprob(logits, target, mask)
    assert acc.ndim == 0 and f1.ndim == 0 and ece.ndim == 0 and lp.shape == target.shape
    foc = masked_focal_loss(logits, target, mask)
    bce = masked_bce_multilabel(torch.randn(2, 4, 3), torch.randint(0, 2, (2, 4, 3)), mask)
    ment = masked_entropy_from_logits(logits, mask)
    assert foc.ndim == 0 and bce.ndim == 0 and ment.ndim == 0


