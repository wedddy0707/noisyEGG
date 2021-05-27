# noisyEGG
## Overview
Source code for my senior thesis.

## Setup

1. clone this repo and move to the directory.
```
$ git clone --recursive https://github.com/wedddy0707/noisyEGG.git
$ cd noisyEGG
```

2. move to EGG directory, git checkout, and install EGG.
```
$ cd EGG
$ git checkout v1.0
$ pip install --editable .
$ cd -  # back to the previous directory
```

## Training
```
$ cd noisy_env_soft_maxlen
$ python train.py [options]
```
### Some notable options
1. Noises
```
--sender_noise_loc      # mean of Gaussian noise on a speaker (sender)
--sender_noise_scale    # standard deviation of Gaussian noise on a speaker (sender)
--receiver_noise_loc    # mean of Gaussian noise on a listener (receiver)
--receiver_noise_scale  # standard deviation of Gaussian noise on a listener (receiver)
--channel_repl_prob     # channel replacement probability
```
2. DER (Decayed Entropy Regularizer)
```
--sender_entropy_coeff         # $\lambda_{\mathcal{H}}$ in the paper
--sender_entropy_common_ratio  # $\rho_{\mathcal{H}}$ in the paper
```
3. SML (Soft Max Len)
```
--length_cost        # $\lambda_{sml}$ in the paper
--max_len            # ${\rm max_len}$ in the paper
--effective_max_len  # ${\rm eff_max_len}$ in the paper
```

For more information, please refer to EGG.
