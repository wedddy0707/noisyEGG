# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# https://github.com/facebookresearch/EGG/blob/master/LICENSE

from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from egg.core.util import find_lengths
from egg.core.baselines import MeanBaseline

from rnn import RnnEncoder


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
        self.agent = agent

        self.force_eos = force_eos

        self.max_len = max_len
        if force_eos:
            assert self.max_len > 1, "Cannot force eos when max_len is below 1"
            self.max_len -= 1

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.cells = None

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
        prev_hidden = [self.agent(x)]
        prev_hidden.extend([torch.zeros_like(prev_hidden[0])
                            for _ in range(self.num_layers - 1)])

        prev_c = [
            torch.zeros_like(
                prev_hidden[0]) for _ in range(
                self.num_layers)]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * x.size(0))

        sequence = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells):
                if self.training:
                    if isinstance(layer, nn.LSTMCell):
                        e = torch.randn_like(prev_c[i]).to(prev_c[i].device)
                        prev_c[i] = (
                            prev_c[i] + self.noise_loc + e * self.noise_scale
                        )
                    else:
                        e = torch.randn_like(
                            prev_hidden[i]).to(
                            prev_hidden[i].device)
                        prev_hidden[i] = (
                            prev_hidden[i] + self.noise_loc + e * self.noise_scale)

                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])
                prev_hidden[i] = h_t
                input = h_t

            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(x))

            input = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy


class RnnReceiverDeterministic(nn.Module):
    def __init__(self,
                 agent,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 cell='rnn',
                 num_layers=1,
                 noise_loc=0.0,
                 noise_scale=0.0,
                 ):
        super(RnnReceiverDeterministic, self).__init__()
        self.agent = agent
        self.encoder = RnnEncoder(
            vocab_size,
            embed_dim,
            hidden_size,
            cell,
            num_layers,
            noise_loc=noise_loc,
            noise_scale=noise_scale,
        )

    def forward(self, message, input=None, lengths=None):
        encoded = self.encoder(message)
        agent_output = self.agent(encoded, input)

        logits = torch.zeros(agent_output.size(0)).to(agent_output.device)
        entropy = logits

        return agent_output, logits, entropy


class SenderReceiverRnnReinforce(nn.Module):
    def __init__(self,
                 sender,
                 receiver,
                 loss,
                 sender_entropy_coeff,
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
        self.sender_entropy_coeff = sender_entropy_coeff
        self.sender_entropy_common_ratio = sender_entropy_common_ratio
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.length_cost = length_cost
        self.machineguntalk_cost = machineguntalk_cost

        self.channel = channel

        self.baselines = defaultdict(baseline_type)

    def forward(self, sender_input, labels, receiver_input=None):
        message, log_prob_s, entropy_s = self.sender(sender_input)

        if self.training:
            message = self.channel(message)

        message_lengths = find_lengths(message)
        receiver_output, log_prob_r, entropy_r = self.receiver(
            message, receiver_input, message_lengths)

        loss, rest = self.loss(
            sender_input, message, receiver_input, receiver_output, labels)

        # the entropy of the outputs of S before and including the eos symbol -
        # as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r)

        # decayed_ratio:
        #   the ratio of sender entropy's weight at each time step
        # decayed_denom:
        #   the denominator for the weighted mean of sender entropy
        decayed_ratio = 1.0
        decayed_denom = torch.zeros_like(message_lengths).float()
        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * decayed_ratio * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
            decayed_denom += decayed_ratio * not_eosed
            # update decayed_ratio geometrically
            decayed_ratio = decayed_ratio * self.sender_entropy_common_ratio
        effective_entropy_s = effective_entropy_s / decayed_denom

        weighted_entropy = effective_entropy_s.mean() * self.sender_entropy_coeff + \
            entropy_r.mean() * self.receiver_entropy_coeff

        log_prob = effective_log_prob_s + log_prob_r

        length_loss = message_lengths.float() * self.length_cost

        policy_length_loss = (
            (length_loss -
             self.baselines['length'].predict(length_loss)) *
            effective_log_prob_s).mean()
        policy_loss = (
            (loss.detach() -
             self.baselines['loss'].predict(
                loss.detach())) *
            log_prob).mean()

        # my new loss
        machineguntalk_loss = (
            (message_lengths == message.size(1)).float() *
            self.machineguntalk_cost)
        policy_machineguntalk_loss = (
            (machineguntalk_loss -
             self.baselines['machineguntalk'].predict(machineguntalk_loss)) *
            effective_log_prob_s).mean()

        optimized_loss = (
            policy_machineguntalk_loss +
            policy_length_loss +
            policy_loss -
            weighted_entropy
        )
        # if the receiver is deterministic/differentiable, we apply the actual
        # loss
        optimized_loss += loss.mean()

        if self.training:
            self.baselines['loss'].update(loss)
            self.baselines['length'].update(length_loss)
            self.baselines['machineguntalk'].update(machineguntalk_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = message_lengths.float().mean().item()

        return optimized_loss, rest
