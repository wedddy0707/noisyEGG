# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# https://github.com/facebookresearch/EGG/blob/master/LICENSE


from collections import defaultdict
import torch.nn as nn
import torch

from egg.core.baselines import MeanBaseline
from egg.core.util import find_lengths


class SenderReceiverRnnReinforce(nn.Module):
    def __init__(self,
                 sender,
                 receiver,
                 loss,
                 sender_entropy_coeff,
                 receiver_entropy_coeff,
                 length_cost=0.0,
                 effective_max_len=None,
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
        self.effective_max_len = effective_max_len

        self.channel = channel

        self.baselines = defaultdict(baseline_type)

    def forward(self, sender_input, labels, receiver_input=None):
        ##############################
        # Sender Forward Propagation #
        ##############################
        message, logprob_s, entropy_s = self.sender(sender_input)

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
        if self.effective_max_len is None:
            len_loss = torch.zeros_like(lengths).to(lengths.device).float()
        else:
            len_loss = self.len_cost * (
                lengths - self.effective_max_len
            ).clamp_(min=0).float()

        policy_loss = ((
            loss.detach() - self.baselines['loss'].predict(loss.detach())
        ) * logprob).mean()
        policy_len_loss = ((
            len_loss - self.baselines['len'].predict(len_loss)
        ) * effective_logprob_s).mean()

        optimized_loss = (
            loss.mean() +
            policy_loss +
            policy_len_loss -
            entropy
        )

        if self.training:
            self.baselines['loss'].update(loss)
            self.baselines['len'].update(len_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = lengths.float().mean().item()

        return optimized_loss, rest
