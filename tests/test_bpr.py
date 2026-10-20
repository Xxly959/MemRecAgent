"""Guard BPR's sign, batch reduction, stability and three-vector update."""

import pytest
import torch

from memrecagent import bpr


def test_loss_and_score_gradients_match_two_hand_calculated_pairs():
    positive = torch.tensor([4.0, 1.0], requires_grad=True)
    negative = torch.tensor([2.0, 3.0], requires_grad=True)
    loss = bpr.bpr_loss(positive, negative)
    assert loss.item() == pytest.approx(1.126928011)
    loss.backward()
    torch.testing.assert_close(positive.grad, torch.tensor([-0.059601461, -0.440398539]))
    torch.testing.assert_close(negative.grad, torch.tensor([0.059601461, 0.440398539]))


def test_larger_positive_lead_reduces_loss():
    behind = bpr.bpr_loss(torch.tensor([2.0]), torch.tensor([3.0]))
    ahead = bpr.bpr_loss(torch.tensor([4.0]), torch.tensor([3.0]))
    assert behind.item() == pytest.approx(1.313261688)
    assert ahead.item() == pytest.approx(0.313261688)
    assert ahead < behind


def test_adding_same_offset_to_both_scores_preserves_pairwise_loss():
    first = bpr.bpr_loss(torch.tensor([4.0]), torch.tensor([3.0]))
    shifted = bpr.bpr_loss(torch.tensor([9.0]), torch.tensor([8.0]))
    torch.testing.assert_close(first, shifted)


def test_extreme_scores_have_finite_loss_and_gradients():
    positive = torch.tensor([-1000.0, 1000.0], requires_grad=True)
    negative = torch.tensor([1000.0, -1000.0], requires_grad=True)
    loss = bpr.bpr_loss(positive, negative)
    assert loss.item() == 1000.0
    loss.backward()
    assert torch.isfinite(positive.grad).all()
    assert torch.isfinite(negative.grad).all()


@pytest.mark.parametrize("positive,negative", [([], []), ([1.0, 2.0], [1.0]), ([[1.0]], [[2.0]])])
def test_loss_rejects_empty_or_broadcasting_score_batches(positive, negative):
    with pytest.raises(ValueError):
        bpr.bpr_loss(torch.tensor(positive), torch.tensor(negative))
