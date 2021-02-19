# Copyright (c) 2021 Ryo Ueda

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def prefix_test(game, n_features, device, add_eos=False):
    train_state = game.training  # persist so we restore it back
    game.eval()

    with torch.no_grad():
        inputs = torch.eye(n_features).to(device)
        messages = game.sender(inputs)
        messages = messages[0]
        max_len = messages.size(1)
        if add_eos:
            max_len += 1

        for i, m in zip(inputs, messages):
            for m_idx in range(max_len):
                if add_eos:
                    prefix = torch.cat(
                        (m[0:m_idx], torch.tensor([0]).to(m.device))
                    )
                    prefix_type = 'prefix_with_eos'
                else:
                    prefix = m[0:m_idx + 1]
                    prefix_type = 'prefix_without_eos'
                o = game.receiver(torch.stack([prefix]))
                o = o[0]

                comma_separeted_prefix = ",".join([
                    str(prefix[i].item()) for i in range(prefix.size(0))
                ])
                dump_message = (
                    f'input: {i.argmax().item()} -> '
                    f'{prefix_type}: {comma_separeted_prefix} -> '
                    f'output: {o.argmax().item()}'
                )
                print(dump_message, flush=True)

                if m[min(m_idx, m.size(0) - 1)] == 0:
                    break

    game.train(mode=train_state)


def first_eos_index(x):
    for i in range(x.size(0)):
        if x[0] == 0:
            return i
    return x.size(0) - 1


def suffix_test(game, n_features, device):
    train_state = game.training  # persist so we restore it back
    game.eval()

    with torch.no_grad():
        inputs = torch.eye(n_features).to(device)
        messages = game.sender(inputs)
        messages = messages[0]
        max_len = messages.size(1)
        for i, m in zip(inputs, messages):
            for m_idx in range(max_len):
                suffix = m[m_idx:]
                suffix = suffix[:first_eos_index(suffix) + 1]
                o = game.receiver(torch.stack([suffix]))
                o = o[0]

                comma_separeted_suffix = ",".join([
                    str(suffix[i].item()) for i in range(suffix.size(0))
                ])
                dump_message = (
                    f'input: {i.argmax().item()} -> '
                    f'suffix: {comma_separeted_suffix} -> '
                    f'output: {o.argmax().item()}'
                )
                print(dump_message, flush=True)

                if m[m_idx] == 0:
                    break

    game.train(mode=train_state)
