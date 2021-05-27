# Copyright (c) 2021 Ryo Ueda

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn


class GaussNoise(nn.Module):
    def __init__(self, loc, scale):
        super(GaussNoise, self).__init__()
        self.loc = loc if loc is not None else 0.0
        self.scale = scale if scale is not None else 0.0

    def forward(self, x):
        return x + float(self.training) * (
            self.loc + self.scale * torch.randn_like(x).to(x.device)
        )


class Noise(nn.Module):
    def __init__(
        self,
        loc=None,
        scale=None,
        dropout_p=None,
    ):
        super(Noise, self).__init__()
        if dropout_p is not None:
            self.layer = nn.Dropout(p=dropout_p)
        else:
            self.layer = GaussNoise(loc=loc, scale=scale)

    def forward(self, x):
        return self.layer(x)
