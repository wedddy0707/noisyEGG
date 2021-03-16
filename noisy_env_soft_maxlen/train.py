# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in
# https://github.com/facebookresearch/EGG/blob/master/LICENSE

import argparse
import numpy as np

import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.zoo.channel.features import OneHotLoader, UniformLoader
from egg.zoo.channel.archs import Sender, Receiver
from egg.zoo.channel.train import loss, dump


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from common import Channel                     # noqa: E402
from common import RnnSenderReinforce          # noqa: E402
from common import RnnReceiverDeterministic    # noqa: E402
from common import SenderReceiverRnnReinforce  # noqa: E402
from common import prefix_test                 # noqa: E402
from common import suffix_test                 # noqa: E402
from common import replacement_test            # noqa: E402
from common import get_common_params           # noqa: E402


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sender_noise_loc',
        type=float,
        default=0.0,
        help='The mean of the noise '
        'added to the hidden layers of Sender')
    parser.add_argument(
        '--sender_noise_scale',
        type=float,
        default=0.0,
        help='The standard deviation of the noise '
        'added to the hidden layers of Sender')
    parser.add_argument(
        '--receiver_noise_loc',
        type=float,
        default=0.0,
        help='The mean of the noise '
        'added to the hidden layers of Receiver')
    parser.add_argument(
        '--receiver_noise_scale',
        type=float,
        default=0.0,
        help='The standard deviation of the noise '
        'added to the hidden layers of Receiver')
    parser.add_argument(
        '--channel_repl_prob',
        type=float,
        default=0.0,
        help="The probability of peplacement of each signal")
    parser.add_argument(
        '--sender_entropy_common_ratio',
        type=float,
        default=1.0,
        help="the common_ratio of the weights of sender entropy")
    parser.add_argument(
        '--effective_max_len',
        type=int,
        default=None,
        help='effective max len'
    )
    parser.add_argument(
        '--sender_dropout_p',
        type=float,
        default=None,
        help='sender dropout p',
    )
    parser.add_argument(
        '--receiver_dropout_p',
        type=float,
        default=None,
        help='receiver dropout p',
    )

    return get_common_params(params, parser)


def main(params):
    opts = get_params(params)
    print(opts, flush=True)
    device = opts.device

    force_eos = opts.force_eos == 1

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
        force_eos=force_eos,
        noise_loc=opts.sender_noise_loc,
        noise_scale=opts.sender_noise_scale,
        dropout_p=opts.sender_dropout_p)

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
        noise_scale=opts.receiver_noise_scale,
        dropout_p=opts.receiver_dropout_p)

    ###################
    # define  channel #
    ###################
    channel = Channel(vocab_size=opts.vocab_size, p=opts.channel_repl_prob)

    game = SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=opts.receiver_entropy_coeff,
        length_cost=opts.length_cost,
        effective_max_len=opts.effective_max_len,
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
            - sender_entropy_coeff as sentr
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
            f'_sentr{opts.sender_entropy_coeff}' +
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

    if opts.checkpoint_path_to_evaluate is None:
        trainer.train(n_epochs=opts.n_epochs)
    else:
        trainer.load_from_checkpoint(opts.checkpoint_path_to_evaluate)
    print('<div id="prefix test without eos">')
    prefix_test(trainer.game, opts.n_features, device, add_eos=False)
    print('</div>')
    print('<div id="prefix test with eos">')
    prefix_test(trainer.game, opts.n_features, device, add_eos=True)
    print('<div id="suffix test">')
    suffix_test(trainer.game, opts.n_features, device)
    print('</div>')
    print('<div id="replacement test">')
    replacement_test(trainer.game, opts.n_features, opts.vocab_size, device)
    print('</div>')
    print('<div id="dump">')
    dump(trainer.game, opts.n_features, device, False)
    print('</div>')
    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
