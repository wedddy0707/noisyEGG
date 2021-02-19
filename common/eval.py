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
        if x[i] == 0:
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


def replacement_test(game, n_features, vocab_size, device):
    train_state = game.training  # persist so we restore it back
    game.eval()

    with torch.no_grad():
        inputs = torch.eye(n_features).to(device)
        messages = game.sender(inputs)
        messages = messages[0]
        max_len = messages.size(1)
        for i, m in zip(inputs, messages):
            i_symb = i.argmax().item()
            all_failure_cnt = 0
            eosed = False
            for m_idx in range(max_len):
                if m[m_idx] == 0:
                    eosed = True
                    break
                each_failure_cnt = 0
                for dummy_symb in range(1, vocab_size):
                    if m[m_idx].item() == dummy_symb:
                        continue
                    m_repl = torch.cat([
                        m[:m_idx],
                        torch.tensor([dummy_symb]).to(device),
                        m[m_idx + 1:first_eos_index(m) + 1]
                    ])
                    o = game.receiver(torch.stack([m_repl]))
                    o = o[0]
                    o_symb = o.argmax().item()
                    each_failure_cnt += int(not i_symb == o_symb)

                    comma_separeted_message = ",".join([
                        str(m_repl[i].item()) for i in range(m_repl.size(0))
                    ])
                    dump_message = (
                        f'input: {i_symb} -> '
                        f'replaced_at{m_idx}_to{dummy_symb}: '
                        f'{comma_separeted_message} -> '
                        f'output: {o_symb}'
                    )
                    print(dump_message, flush=True)
                effectiveness = each_failure_cnt / (n_features - 2)
                print(
                    f'input: {i_symb} -> '
                    f'effectiveness_of_idx{m_idx}: {effectiveness}'
                )
                all_failure_cnt += each_failure_cnt
            effective_length = all_failure_cnt / (n_features - 2)
            if eosed:
                effective_length += 1
            print(
                f'input: {i_symb} -> '
                f'effective_length_by_replacement: {effective_length}'
            )

    game.train(mode=train_state)
