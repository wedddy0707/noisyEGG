# Copyright (c) 2021 Ryo Ueda

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# This file is based on egg.core.reingorce_wrappers.py


from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

from egg.core.baselines import MeanBaseline

from util import find_lengths  # Note that this 'utils' is not the file in EGG


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
            noise_loc=0.0,
            noise_scale=0.0,
    ):
        super(RnnSenderReinforce, self).__init__()
        self.agent = agent
        self.max_len = max_len

        self.output_symbol = nn.Linear(hidden_size, vocab_size)
        self.whether_to_stop = nn.Linear(hidden_size, 1)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.noise_loc = noise_loc
        self.noise_scale = noise_scale

        cell = cell.lower()
        cell_types = {
            'rnn': nn.RNNCell,
            'gru': nn.GRUCell,
            'lstm': nn.LSTMCell}

        if cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        cell_type = cell_types[cell]
        self.cells = nn.ModuleList(
            [
                cell_type(
                    input_size=embed_dim,
                    hidden_size=hidden_size) if i == 0 else cell_type(
                    input_size=hidden_size,
                    hidden_size=hidden_size) for i in range(
                    self.num_layers)])

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

        symb_seq = []
        stop_seq = []
        symb_logits = []
        stop_logits = []
        symb_entropy = []
        stop_entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells):
                e_t = float(self.training) * (
                    self.noise_loc +
                    self.noise_scale * torch.randn_like(prev_h[0]).to(prev_h[0])
                )
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_h[i], prev_c[i]))
                    c_t = c_t + e_t
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_h[i])
                    h_t = h_t + e_t
                prev_h[i] = h_t
                input = h_t

            symb_probs = F.softmax(self.output_symbol(h_t), dim=1)
            stop_probs = torch.sigmoid(
                torch.squeeze(self.whether_to_stop(h_t), 1)
            )
            symb_distr = Categorical(probs=symb_probs)
            stop_distr = Bernoulli(probs=stop_probs)
            symb = symb_distr.sample() if self.training else symb_probs.argmax(dim=1)
            stop = stop_distr.sample() if self.training else (stop_probs > 0.5).float()
            symb_logits.append(symb_distr.log_prob(symb))
            stop_logits.append(stop_distr.log_prob(stop))
            symb_entropy.append(symb_distr.entropy())
            stop_entropy.append(stop_distr.entropy())
            symb_seq.append(symb)
            stop_seq.append(stop)

            input = self.embedding(symb)

        symb_seq = torch.stack(symb_seq).permute(1, 0)
        stop_seq = torch.stack(stop_seq).permute(1, 0).long()
        symb_logits = torch.stack(symb_logits).permute(1, 0)
        stop_logits = torch.stack(stop_logits).permute(1, 0)
        symb_entropy = torch.stack(symb_entropy).permute(1, 0)
        stop_entropy = torch.stack(stop_entropy).permute(1, 0)

        logits = (symb_logits, stop_logits)
        entropy = (symb_entropy, stop_entropy)

        return symb_seq, stop_seq, logits, entropy


class SenderReceiverRnnReinforce(nn.Module):
    def __init__(self,
                 sender,
                 receiver,
                 loss,
                 sender_symb_entropy_coeff,
                 sender_stop_entropy_coeff,
                 receiver_entropy_coeff,
                 length_cost=0.0,
                 machineguntalk_cost=0.0,
                 baseline_type=MeanBaseline,
                 channel=(lambda x: x),
                 sender_entropy_common_ratio=1.0,
                 ):
        super(SenderReceiverRnnReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.sender_symb_entropy_coeff = sender_symb_entropy_coeff
        self.sender_stop_entropy_coeff = sender_stop_entropy_coeff
        self.sender_entropy_common_ratio = sender_entropy_common_ratio
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.len_cost = length_cost
        self.mgt_cost = machineguntalk_cost

        self.channel = channel

        self.baselines = defaultdict(baseline_type)

    def forward(self, sender_input, labels, receiver_input=None):
        ######################################
        # Forward Propagation through Sender #
        ######################################
        symb_seq_s, stop_seq_s, logprob_s, entropy_s = self.sender(sender_input)
        symb_logprob_s, stop_logprob_s = logprob_s
        symb_entropy_s, stop_entropy_s = entropy_s

        #######################################
        # Forward Propagation through Channel #
        #######################################
        if self.training:
            symb_seq_s = self.channel(symb_seq_s)

        #########################################
        # Check lengths of sequences (messages) #
        #########################################
        lengths = find_lengths(stop_seq_s)
        max_len = symb_seq_s.size(1)

        ########################################
        # Forward Propagation through Receiver #
        ########################################
        receiver_output, logprob_r, entropy_r = self.receiver(
            symb_seq_s,
            receiver_input,
            lengths)

        ############################################
        # Calculation of Effective logprob/entropy #
        ############################################
        eff_symb_entropy_s = torch.zeros_like(entropy_r)
        eff_stop_entropy_s = torch.zeros_like(entropy_r)
        eff_logprob_s = torch.zeros_like(logprob_r)
        denom = torch.zeros_like(lengths).float()
        ratio = 1.0
        for i in range(symb_seq_s.size(1)):
            not_stopped = (i < lengths).float()
            eff_symb_entropy_s += symb_entropy_s[:, i] * not_stopped * ratio
            eff_stop_entropy_s += stop_entropy_s[:, i] * not_stopped * ratio
            eff_logprob_s += symb_logprob_s[:, i] * not_stopped
            eff_logprob_s += stop_logprob_s[:, i] * not_stopped
            denom += ratio * not_stopped
            ratio *= self.sender_entropy_common_ratio
        eff_symb_entropy_s = eff_symb_entropy_s / denom
        eff_stop_entropy_s = eff_stop_entropy_s / denom

        logprob = eff_logprob_s + logprob_r
        entropy = (
            eff_symb_entropy_s.mean() * self.sender_symb_entropy_coeff +
            eff_stop_entropy_s.mean() * self.sender_stop_entropy_coeff +
            entropy_r.mean() * self.receiver_entropy_coeff
        )

        #######################
        # Calculation of Loss #
        #######################
        loss, rest = self.loss(
            sender_input, symb_seq_s, receiver_input, receiver_output, labels)
        # Auxiliary losses
        len_loss = self.len_cost * lengths.float()
        mgt_loss = self.mgt_cost * (lengths == max_len).float()

        policy_loss = loss.detach() - \
            self.baselines['loss'].predict(loss.detach())
        policy_loss = (policy_loss * logprob).mean()
        policy_len_loss = len_loss - self.baselines['len'].predict(len_loss)
        policy_len_loss = (policy_len_loss * eff_logprob_s).mean()
        policy_mgt_loss = mgt_loss - self.baselines['mgt'].predict(mgt_loss)
        policy_mgt_loss = (policy_mgt_loss * eff_logprob_s).mean()

        optimized_loss = (
            loss.mean() +
            policy_mgt_loss +
            policy_len_loss +
            policy_loss -
            entropy
        )

        if self.training:
            self.baselines['loss'].update(loss)
            self.baselines['len'].update(len_loss)
            self.baselines['mgt'].update(mgt_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = symb_entropy_s.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = lengths.float().mean().item()

        return optimized_loss, rest
