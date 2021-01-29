# Copyright (c) 2021 Ryo Ueda

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


class Channel():
    def __init__(self, vocab_size, p=0.0):
        if vocab_size < 3:
            # no replacement will occure
            self.p = 0.0
        else:
            self.p = p * float(vocab_size - 1) / float(vocab_size - 2)
        self.vocab_size = vocab_size

    def __call__(self, message: torch.Tensor):
        p = torch.full_like(message, self.p, dtype=torch.double)

        repl_choice = torch.bernoulli(p) == 1.0
        repl_value = torch.randint_like(message, 1, self.vocab_size)

        inv_zero_mask = ~(message == 0)

        message = (
            message + inv_zero_mask * repl_choice * (repl_value - message)
        )
        return message
