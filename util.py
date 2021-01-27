import torch


def find_lengths(cancellation: torch.Tensor) -> torch.Tensor:
    max_k = cancellation.size(1)
    lengths = max_k - (cancellation.cumsum(dim=1) > 0).sum(dim=1)
    lengths.add_(1).clamp_(max=max_k)

    return lengths
