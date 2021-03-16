# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# https://github.com/facebookresearch/EGG/blob/master/LICENSE

from typing import Optional

import torch
import torch.nn as nn

from egg.core.util import find_lengths

from .noise import Noise


class NoisyCell(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_hidden: int,
        cell: str = "rnn",
        num_layers: int = 1,
        noise_loc=None,
        noise_scale=None,
        dropout_p=None,
    ) -> None:
        super(NoisyCell, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = n_hidden

        cell = cell.lower()
        cell_type = {"rnn": nn.RNNCell, "gru": nn.GRUCell, "lstm": nn.LSTMCell}

        if cell not in cell_type:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.isLSTM = cell == "lstm"

        self.noise_layer = Noise(
            loc=noise_loc,
            scale=noise_scale,
            dropout_p=dropout_p,
        )

        self.cells = nn.ModuleList([
            cell_type[cell](embed_dim, n_hidden) if i == 0 else
            cell_type[cell](n_hidden, n_hidden) for i in range(num_layers)])

    def forward(self, input: torch.Tensor, h_0: Optional[torch.Tensor] = None):
        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0].item()
            device = input.device
            if h_0 is None:
                prev_h = [
                    torch.zeros(max_batch_size, self.hidden_size).to(device)
                    for _ in range(self.num_layers)
                ]
                prev_c = [
                    torch.zeros_like(prev_h[0])
                    for _ in range(self.num_layers)
                ]
            elif self.isLSTM:
                h_0, c_0 = h_0
                prev_h = [x for x in h_0.permute(1, 0, 2)]
                prev_c = [x for x in c_0.permute(1, 0, 2)]
            else:
                prev_h = [x for x in h_0.permute(1, 0, 2)]

            input_idx = 0
            for batch_size in batch_sizes.tolist():
                x = input[input_idx:input_idx + batch_size]
                for i, layer in enumerate(self.cells):
                    if self.isLSTM:
                        h, c = layer(x, (prev_h[i][:batch_size], prev_c[i][:batch_size]))  # noqa: E501
                        c = self.noise_layer(c)
                        prev_c[i] = torch.cat((c, prev_c[i][batch_size:]))  # noqa: E501
                    else:
                        h = layer(x, prev_h[i][:batch_size])
                        h = self.noise_layer(h)
                    prev_h[i] = torch.cat((h, prev_h[i][batch_size:]))
                    x = h
                input_idx += batch_size

            h = prev_h
            h = torch.stack(h)
            h = torch.index_select(h, 1, unsorted_indices)
            if self.isLSTM:
                c = prev_c
                c = torch.stack(c)
                c = torch.index_select(c, 1, unsorted_indices)
                h = (h, c)
        else:
            # only implemented for packed_sequence
            pass
        output = None  # output is not implemented
        return output, h


class RnnEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_hidden: int,
        cell: str = 'rnn',
        num_layers: int = 1,
        noise_loc=None,
        noise_scale=None,
        dropout_p=None,
    ) -> None:
        super(RnnEncoder, self).__init__()

        self.noisycell = NoisyCell(
            embed_dim,
            n_hidden,
            cell,
            num_layers,
            noise_loc,
            noise_scale,
            dropout_p,
        )

        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(
        self,
        message: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        h_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.embedding(message)

        if lengths is None:
            lengths = find_lengths(message)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, rnn_hidden = self.noisycell(packed, h_0)

        if self.noisycell.isLSTM:
            rnn_hidden, _ = rnn_hidden

        return rnn_hidden[-1]
