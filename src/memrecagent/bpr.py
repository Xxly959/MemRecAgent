"""Pairwise ranking loss; sampling and embedding regularization are separate."""

import torch
import torch.nn.functional as F


def bpr_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    """Average -log(sigmoid(positive-negative)) over a nonempty paired batch.

    Inputs are finite floating-point scores, not star labels. softplus avoids
    explicitly exponentiating large margins. This does not certify that a
    sampled unobserved item is truly disliked, nor directly optimize Recall@K.
    """
    if (
        positive_scores.ndim != 1
        or negative_scores.ndim != 1
        or positive_scores.shape != negative_scores.shape
        or positive_scores.numel() == 0
    ):
        raise ValueError("Scores must be matching nonempty one-dimensional batches")
    return F.softplus(negative_scores - positive_scores).mean()
