# Copyright (c) 2021 Ryo Ueda

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# This file is based on egg.core.util.py


import torch
import numpy as np
from typing import Optional


def find_lengths(
        stop_seq: torch.Tensor,
        stop_logprob: Optional[torch.Tensor] = None,
        mode='naive',
) -> torch.Tensor:
    max_len = stop_seq.size(1)
    if mode == 'naive':
        lengths = max_len - (stop_seq.cumsum(dim=1) > 0).sum(dim=1)
    elif mode == 'expectation' and stop_logprob is not None:
        stop_seq = stop_seq.float()
        logprob_to_stop = (
            stop_seq * stop_logprob +
            (1. - stop_seq) * torch.log(1. - torch.exp(stop_logprob))
        )
        logprob_not_to_stop = (
            (1. - stop_seq) * stop_logprob +
            stop_seq * torch.log(1. - torch.exp(stop_logprob))
        )
        lengths = torch.zeros_like(stop_logprob)
        for i in range(max_len):
            lengths[:, i] += np.log(float(i))
            lengths[:, i] += logprob_to_stop[:, i]
            for j in range(i):
                lengths[:, i] += logprob_not_to_stop[:, j]
        lengths = torch.exp(lengths)
        lengths = lengths.mean(dim=1).round().long()

    lengths.add_(1).clamp_(max=max_len)
    return lengths
