# noisyEGG

## 概要
卒論で用いたコードを，使いやすいように整理する．

## フォルダ
noisy\_env
    ~ 卒論の実験を再現するためのコードです．

noisy\_env\_eos\_separated
    ~ 卒論には間に合わなかったものの，追加で行う実験のためのコードです．

## 使い方

EGG を予めインストールしておいてください．
インストールの手順については，https://github.com/facebookresearch/EGG を参照してください．

### 実行方法
EGG のバージョンを切り替えます．
EGG のフォルダに移動して，以下のコマンドを実行し，tag v1.0 を使うようにします．
```
$ git checkout v1.0
```

このフォルダに戻り，以下のコマンドを実行すると学習が始まります．
```
$ python train.py [options]
```

### 主なオプション
#### 基本的な学習方法にかかわる部分
```
--n_epoch              # epoch 数
--n_batches_per_epoch  # epoch 毎の batch の数
--batch_size           # batch のサイズ
--validation_freq      # バリデーションの頻度
--lr                   # 学習率
--early_stopping_thr   # 早期終了の精度
```

#### loss function に関わる部分
```
--sender_entropy_coeff         # (decayed) entropy regularizer の係数
--sender_entropy_common_ratio  # (decayed) entropy regularizer の減衰率
--length_cost                  # length pressure の係数
--machineguntalk_cost          # machine-gun-talk penalty の係数
```

#### シグナリング・ゲームの設定
```
--vocab_size  # vocabulary (alphabet) のサイズ
--n_features  # input space のサイズ
--probs       # input のサンプル元となる分布．power-law or uniform.
--max_len     # message の最大長
```

#### アーキテクチャの設定
```
--sender_cell         # sender (speaker) の基本構造 rnn, lstm, or gru.
--sender_hidden       # sender (speaker) の隠れ層のサイズ
--sender_embedding    # sender (speaker) の埋め込み層のサイズ
--receiver_cell       # receiver (listener) の基本構造 rnn, lstm, or gru.
--receiver_hidden     # receiver (listener) の隠れ層のサイズ
--receiver_embedding  # receiver (listener) の埋め込み層ののサイズ
```

#### ノイズの設定
```
--sender_noise_loc      # sender ノイズの平均
--sender_noise_scale    # sender ノイズの標準偏差
--receiver_noise_loc    # receiver ノイズの平均
--receiver_noise_scale  # receiver ノイズの標準偏差
--channel_repl_prob     # channel replacement probability
```
