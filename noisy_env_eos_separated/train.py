# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in
# https://github.com/facebookresearch/EGG/blob/master/LICENSE

import json
import argparse
import numpy as np
import torch.utils.data

import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.zoo.channel.features import OneHotLoader, UniformLoader
from egg.zoo.channel.archs import Sender, Receiver
from egg.zoo.channel.train import loss

from util import find_lengths
from reinforce_wrappers import RnnSenderReinforce
from reinforce_wrappers import SenderReceiverRnnReinforce

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from common import Channel                     # noqa: E402
from common import RnnReceiverDeterministic    # noqa: E402


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_features',
        type=int,
        default=10,
        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument('--batches_per_epoch', type=int, default=1000,
                        help='Number of batches per epoch (default: 1000)')
    parser.add_argument(
        '--force_eos',
        type=int,
        default=0,
        help='Force EOS at the end of the messages (default: 0)')

    parser.add_argument(
        '--sender_hidden',
        type=int,
        default=10,
        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument(
        '--receiver_hidden',
        type=int,
        default=10,
        help='Size of the hidden layer of Receiver (default: 10)')
    parser.add_argument(
        '--receiver_num_layers',
        type=int,
        default=1,
        help='Number hidden layers of receiver. Only in reinforce (default: 1)')
    parser.add_argument(
        '--sender_num_layers',
        type=int,
        default=1,
        help='Number hidden layers of receiver. Only in reinforce (default: 1)')
    parser.add_argument(
        '--sender_embedding',
        type=int,
        default=10,
        help='Dimensionality of the embedding hidden layer for Sender (default: 10)')
    parser.add_argument(
        '--receiver_embedding',
        type=int,
        default=10,
        help='Dimensionality of the embedding hidden layer for Receiver (default: 10)')

    parser.add_argument(
        '--sender_cell',
        type=str,
        default='rnn',
        help='Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)')
    parser.add_argument(
        '--receiver_cell',
        type=str,
        default='rnn',
        help='Type of the model used for Receiver {rnn, gru, lstm} (default: rnn)')
    parser.add_argument(
        '--sender_symb_entropy_coeff',
        type=float,
        default=1e-1,
        help='The entropy regularisation coefficient for Sender\'s symbol distr (default: 1e-1)')
    parser.add_argument(
        '--sender_stop_entropy_coeff',
        type=float,
        default=1e-1,
        help='The entropy regularisation coefficient for Sender\' stop distr (default: 1e-1)')
    parser.add_argument(
        '--receiver_entropy_coeff',
        type=float,
        default=1e-1,
        help='The entropy regularisation coefficient for Receiver (default: 1e-1)')

    parser.add_argument(
        '--probs',
        type=str,
        default='uniform',
        help="Prior distribution over the concepts (default: uniform)")
    parser.add_argument(
        '--length_cost',
        type=float,
        default=0.0,
        help="Penalty for the message length, each symbol would before <EOS> would be "
        "penalized by this cost (default: 0.0)")
    parser.add_argument('--name', type=str, default='model',
                        help="Name for your checkpoint (default: model)")
    parser.add_argument(
        '--early_stopping_thr',
        type=float,
        default=0.9999,
        help="Early stopping threshold on accuracy (default: 0.9999)")
    parser.add_argument(
        '--sender_noise_loc',
        type=float,
        default=0.0,
        help="The mean of the noise added to the hidden layers of Sender")
    parser.add_argument(
        '--sender_noise_scale',
        type=float,
        default=0.0,
        help="The standard deviation of the noise added to the hidden layers of Sender")
    parser.add_argument(
        '--receiver_noise_loc',
        type=float,
        default=0.0,
        help="The mean of the noise added to the hidden layers of Receiver")
    parser.add_argument(
        '--receiver_noise_scale',
        type=float,
        default=0.0,
        help="The standard deviation of the noise added to the hidden layers of Receiver")
    parser.add_argument('--channel_repl_prob', type=float, default=0.0,
                        help="The probability of peplacement of each signal")
    parser.add_argument(
        '--sender_entropy_common_ratio',
        type=float,
        default=1.0,
        help="the common_ratio of the weights of sender entropy")
    parser.add_argument(
        '--machineguntalk_cost',
        type=float,
        default=0.0,
        help='the coefficient of machine-gun-talk penalty'
    )

    args = core.init(parser, params)

    return args


def suffix_test(game, n_features, device):
    train_state = game.training  # persist so we restore it back
    game.eval()

    with torch.no_grad():
        inputs = torch.eye(n_features).to(device)
        sender_output = game.sender(inputs)
        messages = sender_output[0]
        stop_seq = sender_output[1]

        for i, m, stop in zip(inputs, messages, stop_seq):
            for m_idx in range(m.size(0)):
                prefix = m[0:m_idx + 1]
                o = game.receiver(torch.stack([prefix]), lengths=torch.tensor([m_idx + 1]))
                o = o[0]

                dump_message = (
                    f'input: {i.argmax().item()} -> '
                    f'prefix: {",".join([str(prefix[i].item()) for i in range(prefix.size(0))])} -> '
                    f'output: {o.argmax().item()}'
                )
                print(dump_message, flush=True)

                if stop[m_idx].item():
                    break

    game.train(mode=train_state)


def dump(game, n_features, device):
    inputs = torch.eye(n_features).to(device)

    train_state = game.training  # persist so we restore it back
    game.eval()

    sender_output = game.sender(inputs)
    messages = sender_output[0]
    stop_seq = sender_output[1]
    lengths = find_lengths(stop_seq)
    outputs, _, _ = game.receiver(messages, None, lengths)

    game.train(mode=train_state)

    uniform_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features + 1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    for i, m, ln, o in zip(inputs, messages, lengths, outputs):
        i_symbol = i.argmax()
        o_symbol = o.argmax()

        is_successful = (i_symbol == o_symbol).float().item()
        uniform_acc += is_successful
        powerlaw_acc += powerlaw_probs[i_symbol] * is_successful

        dump_message = (
            f'input: {i_symbol.item()} -> '
            f'message: {",".join([str(m[i].item()) for i in range(ln.item())])} -> '
            f'output: {o_symbol.item()}'
        )
        print(dump_message, flush=True)

    uniform_acc /= n_features

    print(f'Mean accuracy wrt uniform distribution is {uniform_acc}')
    print(f'Mean accuracy wrt powerlaw distribution is {powerlaw_acc}')
    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': uniform_acc}))


def main(params):
    opts = get_params(params)
    print(opts, flush=True)
    device = opts.device

    if opts.probs == 'uniform':
        probs = np.ones(opts.n_features)
    elif opts.probs == 'powerlaw':
        probs = 1 / np.arange(1, opts.n_features + 1, dtype=np.float32)
    else:
        probs = np.array([float(x)
                          for x in opts.probs.split(',')], dtype=np.float32)
    probs /= probs.sum()

    print('the probs are: ', probs, flush=True)

    train_loader = OneHotLoader(
        n_features=opts.n_features,
        batch_size=opts.batch_size,
        batches_per_epoch=opts.batches_per_epoch,
        probs=probs)

    # single batches with 1s on the diag
    test_loader = UniformLoader(opts.n_features)

    #################################
    # define sender (speaker) agent #
    #################################
    sender = Sender(
        n_features=opts.n_features,
        n_hidden=opts.sender_hidden)
    sender = RnnSenderReinforce(
        sender,
        opts.vocab_size,
        opts.sender_embedding,
        opts.sender_hidden,
        cell=opts.sender_cell,
        max_len=opts.max_len,
        num_layers=opts.sender_num_layers,
        noise_loc=opts.sender_noise_loc,
        noise_scale=opts.sender_noise_scale)

    ####################################
    # define receiver (listener) agent #
    ####################################
    receiver = Receiver(
        n_features=opts.n_features,
        n_hidden=opts.receiver_hidden)
    receiver = RnnReceiverDeterministic(
        receiver,
        opts.vocab_size,
        opts.receiver_embedding,
        opts.receiver_hidden,
        cell=opts.receiver_cell,
        num_layers=opts.receiver_num_layers,
        noise_loc=opts.receiver_noise_loc,
        noise_scale=opts.receiver_noise_scale)

    ###################
    # define  channel #
    ###################
    channel = Channel(vocab_size=opts.vocab_size, p=opts.channel_repl_prob)

    game = SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss,
        sender_symb_entropy_coeff=opts.sender_symb_entropy_coeff,
        sender_stop_entropy_coeff=opts.sender_stop_entropy_coeff,
        receiver_entropy_coeff=opts.receiver_entropy_coeff,
        length_cost=opts.length_cost,
        machineguntalk_cost=opts.machineguntalk_cost,
        channel=channel,
        sender_entropy_common_ratio=opts.sender_entropy_common_ratio
    )

    optimizer = core.build_optimizer(game.parameters())

    callbacks = [EarlyStopperAccuracy(opts.early_stopping_thr),
                 core.ConsoleLogger(as_json=True, print_train_loss=True)]

    if opts.checkpoint_dir:
        '''
        info in checkpoint_name:
            - n_features as f
            - vocab_size as vocab
            - random_seed as rs
            - lr as lr
            - sender_hidden as shid
            - receiver_hidden as rhid
            - sender_symb_entropy_coeff as symbsentr
            - sender_stop_entropy_coeff as stopsentr
            - length_cost as reg
            - max_len as max_len
            - sender_noise_scale as sscl
            - receiver_noise_scale as rscl
            - channel_repl_prob as crp
            - sender_entropy_common_ratio as scr
        '''
        checkpoint_name = (
            f'{opts.name}' +
            ('_uniform' if opts.probs == 'uniform' else '') +
            f'_f{opts.n_features}' +
            f'_vocab{opts.vocab_size}' +
            f'_rs{opts.random_seed}' +
            f'_lr{opts.lr}' +
            f'_shid{opts.sender_hidden}' +
            f'_rhid{opts.receiver_hidden}' +
            f'_symbsentr{opts.sender_symb_entropy_coeff}' +
            f'_stopsentr{opts.sender_stop_entropy_coeff}' +
            f'_reg{opts.length_cost}' +
            f'_max_len{opts.max_len}' +
            f'_sscl{opts.sender_noise_scale}' +
            f'_rscl{opts.receiver_noise_scale}' +
            f'_crp{opts.channel_repl_prob}' +
            f'_scr{opts.sender_entropy_common_ratio}'
        )
        callbacks.append(
            core.CheckpointSaver(
                checkpoint_path=opts.checkpoint_dir,
                checkpoint_freq=opts.checkpoint_freq,
                prefix=checkpoint_name))

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=callbacks)

    trainer.train(n_epochs=opts.n_epochs)

    print('-- suffix test --')
    suffix_test(trainer.game, opts.n_features, device)
    print('-- dump --')
    dump(trainer.game, opts.n_features, device)
    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
