# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# https://github.com/facebookresearch/EGG/blob/master/LICENSE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class RnnSenderReinforce(nn.Module):
    def __init__(
            self,
            agent,
            vocab_size,
            embed_dim,
            hidden_size,
            max_len,
            num_layers=1,
            cell='rnn',
            force_eos=True,
            noise_loc=0.0,
            noise_scale=0.0,
    ):
        super(RnnSenderReinforce, self).__init__()
        assert (not force_eos) or max_len > 1, \
            "Cannot force eos when max_len is below 1"
        self.agent = agent
        self.force_eos = force_eos
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))

        self.noise_loc = noise_loc
        self.noise_scale = noise_scale

        cell = cell.lower()
        cell_types = {
            'rnn': nn.RNNCell,
            'gru': nn.GRUCell,
            'lstm': nn.LSTMCell
        }
        if cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")
        self.cells = nn.ModuleList([
            cell_types[cell](
                input_size=embed_dim,
                hidden_size=hidden_size
            ) if i == 0 else cell_types[cell](
                input_size=hidden_size,
                hidden_size=hidden_size
            ) for i in range(num_layers)
        ])
        self.isLSTM = (cell == 'lstm')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x):
        prev_h = [self.agent(x)]
        prev_h.extend([
            torch.zeros_like(prev_h[0]) for _ in range(self.num_layers - 1)
        ])
        prev_c = [
            torch.zeros_like(prev_h[0]) for _ in range(self.num_layers)
        ]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * x.size(0))

        sequence = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells):
                e_t = float(self.training) * (
                    self.noise_loc +
                    self.noise_scale * torch.randn_like(prev_h[0]).to(prev_h[0])
                )
                if self.isLSTM:
                    h_t, c_t = layer(input, (prev_h[i], prev_c[i]))
                    c_t = c_t + e_t
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_h[i])
                    h_t = h_t + e_t
                prev_h[i] = h_t
                input = h_t

            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            distr = Categorical(logits=step_logits)
            if self.force_eos and step == self.max_len - 1:
                x = torch.tensor([0] * distr.probs.size(0)).to(h_t.device)
            elif self.training:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)
            input = self.embedding(x)
            sequence.append(x)
            logits.append(distr.log_prob(x))
            entropy.append(distr.entropy())

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        return sequence, logits, entropy
