# Copyright (c) 2021 Ryo Ueda

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# This file is based on egg.core.zoo.channel.train.py


import argparse

import egg.core as core


def get_common_params(params, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_features',
        type=int,
        default=10,
        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument(
        '--batches_per_epoch',
        type=int,
        default=1000,
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
        help='Number hidden layers of receiver. '
        'Only in reinforce (default: 1)')
    parser.add_argument(
        '--sender_num_layers',
        type=int,
        default=1,
        help='Number hidden layers of receiver. '
        'Only in reinforce (default: 1)')
    parser.add_argument(
        '--sender_embedding',
        type=int,
        default=10,
        help='Dimensionality of the embedding hidden layer '
        'for Sender (default: 10)')
    parser.add_argument(
        '--receiver_embedding',
        type=int,
        default=10,
        help='Dimensionality of the embedding hidden layer '
        'for Receiver (default: 10)')
    parser.add_argument(
        '--sender_cell',
        type=str,
        default='rnn',
        help='Type of the cell used for Sender '
        '{rnn, gru, lstm} (default: rnn)')
    parser.add_argument(
        '--receiver_cell',
        type=str,
        default='rnn',
        help='Type of the model used for Receiver '
        '{rnn, gru, lstm} (default: rnn)')
    parser.add_argument(
        '--sender_entropy_coeff',
        type=float,
        default=1e-1,
        help='The entropy regularisation coefficient '
        'for Sender (default: 1e-1)')
    parser.add_argument(
        '--receiver_entropy_coeff',
        type=float,
        default=1e-1,
        help='The entropy regularisation coefficient '
        'for Receiver (default: 1e-1)')
    parser.add_argument(
        '--probs',
        type=str,
        default='uniform',
        help="Prior distribution over the concepts (default: uniform)")
    parser.add_argument(
        '--length_cost',
        type=float,
        default=0.0,
        help='Penalty for the message length, '
        'each symbol would before <EOS> would be '
        'penalized by this cost (default: 0.0)')
    parser.add_argument(
        '--name', type=str, default='model',
        help="Name for your checkpoint (default: model)")
    parser.add_argument(
        '--early_stopping_thr',
        type=float,
        default=0.9999,
        help="Early stopping threshold on accuracy (default: 0.9999)")
    parser.add_argument(
        '--checkpoint_path_to_evaluate',
        type=str,
        default=None,
        help=''
    )

    args = core.init(parser, params)

    return args
