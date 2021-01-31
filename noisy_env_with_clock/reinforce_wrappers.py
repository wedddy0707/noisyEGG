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
        self.hidden_to_clock = nn.Linear(hidden_size, 4)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # self.clk_embedding = nn.Embedding(self.max_len, embed_dim)
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

    def sample_symbol_from(self, distr):
        if self.training:
            return distr.sample()
        else:
            return distr.probs.argmax(dim=1)

    def add_noise_to(self, x):
        if self.training:
            e = torch.randn_like(x).to(x)
            x = x + self.noise_loc + e * self.noise_scale
        return x

    def forward(self, x):
        batch_size = x.size(0)
        prev_h = [self.agent(x)]
        prev_h.extend(
            [torch.zeros_like(prev_h[0]) for _ in range(self.num_layers - 1)]
        )
        prev_c = [
            torch.zeros_like(prev_h[0]) for _ in range(self.num_layers)
        ]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * batch_size)

        sequence = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_h[i], prev_c[i]))
                    c_t = self.add_noise_to(c_t)
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_h[i])
                    h_t = self.add_noise_to(h_t)
                prev_h[i] = h_t
                input = h_t

            distr = Categorical(
                logits=F.log_softmax(self.hidden_to_output(h_t), dim=1)
            )
            x = self.sample_symbol_from(distr)
            input = self.embedding(x)
            sequence.append(x)
            logits.append(distr.log_prob(x))
            entropy.append(distr.entropy())

            clk_loss = F.cross_entropy(
                input=self.hidden_to_clock(h_t),
                target=torch.tensor([step] * batch_size).to(x.device) % 4
            )

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)
            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy, clk_loss


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
        self.len_cost = length_cost
        self.mgt_cost = machineguntalk_cost

        self.channel = channel

        self.baselines = defaultdict(baseline_type)

    def forward(self, sender_input, labels, receiver_input=None):
        ##############################
        # Sender Forward Propagation #
        ##############################
        message, logprob_s, entropy_s, aux_loss_s = self.sender(sender_input)

        ###########
        # Channel #
        ###########
        if self.training:
            message = self.channel(message)

        ##########################
        # Calculation of Lengths #
        ##########################
        lengths = find_lengths(message)
        max_len = message.size(1)

        ################################
        # Receiver Forward Propagation #
        ################################
        receiver_output, logprob_r, entropy_r = self.receiver(
            message, receiver_input, lengths)

        ############################################
        # Calculation of Effective Entropy/Logprob #
        ############################################
        effective_entropy_s = torch.zeros_like(entropy_r)
        effective_logprob_s = torch.zeros_like(logprob_r)
        ratio = 1.0
        denom = torch.zeros_like(lengths).float()
        for i in range(max_len):
            not_eosed = (i < lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed * ratio
            effective_logprob_s += logprob_s[:, i] * not_eosed
            denom += ratio * not_eosed
            ratio *= self.sender_entropy_common_ratio
        effective_entropy_s = effective_entropy_s / denom

        logprob = effective_logprob_s + logprob_r
        entropy = (
            effective_entropy_s.mean() * self.sender_entropy_coeff +
            entropy_r.mean() * self.receiver_entropy_coeff
        )

        #######################
        # Calculation of Loss #
        #######################
        loss, rest = self.loss(
            sender_input, message, receiver_input, receiver_output, labels)
        # Auxiliary losses
        len_loss = lengths.float() * self.len_cost
        mgt_loss = (lengths == max_len).float() * self.mgt_cost

        policy_loss = ((
            loss.detach() - self.baselines['loss'].predict(loss.detach())
        ) * logprob).mean()
        policy_len_loss = ((
            len_loss - self.baselines['len'].predict(len_loss)
        ) * effective_logprob_s).mean()
        policy_mgt_loss = ((
            mgt_loss - self.baselines['mgt'].predict(mgt_loss)
        ) * effective_logprob_s).mean()

        optimized_loss = (
            loss.mean() +
            aux_loss_s.mean() +
            policy_loss +
            policy_len_loss +
            policy_mgt_loss -
            entropy
        )

        if self.training:
            self.baselines['loss'].update(loss)
            self.baselines['len'].update(len_loss)
            self.baselines['mgt'].update(mgt_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = lengths.float().mean().item()

        return optimized_loss, rest
