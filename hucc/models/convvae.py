# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from typing import Iterable

import torch as th
import torch.distributions as D
import torch.nn.functional as F
from torch import nn


class ResConv1dBlock(nn.Module):
    def __init__(self, n_in, n_state):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_in, n_state, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(n_state, n_in, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return x + self.model(x)


class Resnet1d(nn.Module):
    def __init__(self, n_in, n_depth):
        super().__init__()
        self.model = nn.Sequential(
            *[ResConv1dBlock(n_in, n_in) for depth in range(n_depth)]
        )

    def forward(self, x):
        return self.model(x)


class ConvVAE(nn.Module):
    def __init__(
        self,
        input_shape: Iterable[int],  # TxC
        block_width: int,
        block_depth: int,
        n_latent: int,
        tanh_latents: bool = False,
    ):
        super().__init__()
        n_in_t, n_in_c = input_shape
        self.latent_dim = n_latent
        self.tanh_latents = tanh_latents

        self.encoder = nn.Sequential(
            nn.Conv1d(n_in_c, 64, kernel_size=3, stride=1, padding='same'),
            Resnet1d(block_width, block_depth),
            nn.Conv1d(
                block_width, n_latent, kernel_size=3, stride=1, padding=1
            ),
        )
        self.bottleneck = nn.Linear(n_latent, n_latent * 2)
        self.decoder = nn.Sequential(
            nn.Conv1d(
                n_latent, block_width, kernel_size=3, stride=1, padding=1
            ),
            Resnet1d(block_width, block_depth),
            nn.ConvTranspose1d(
                block_width, n_latent, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv1d(n_latent, n_in_c, kernel_size=3, stride=1, padding=1),
        )

        self.register_buffer('input_mean', th.zeros(n_in_c))
        self.register_buffer('input_std', th.ones(n_in_c))

    def preprocess(self, x):
        # BTC -> BCT
        assert len(x.shape) == 3
        x = (x - self.input_mean) / self.input_std
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # BCT -> BTC
        x = x.permute(0, 2, 1)
        x = (x * self.input_std) + self.input_mean
        return x

    def encode(self, x, deterministic=None):
        if deterministic is None:
            deterministic = not self.training

        B, T, C = x.shape
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in)
        x_enc = x_enc.permute(0, 2, 1).contiguous()  # BCT -> [BT,C]
        x_enc = x_enc.view(B * T, -1)

        bn_out = self.bottleneck(x_enc)
        z_mu, z_log_std = bn_out.chunk(2, -1)
        z_dist = D.Normal(z_mu, F.softplus(z_log_std))
        if deterministic:
            z = z_dist.mean
        else:
            z = z_dist.rsample()
        if self.tanh_latents:
            z = th.tanh(z)

        z = z.view(B, T, -1)
        return z

    def decode(self, z):
        x_dec = self.decoder(z)
        x_out = self.postprocess(x_dec)
        return x_out

    def forward(self, x):
        B, T, C = x.shape
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in)
        x_enc = x_enc.permute(0, 2, 1).contiguous()  # BCT -> [BT,C]
        x_enc = x_enc.view(B * T, -1)

        bn_out = self.bottleneck(x_enc)
        z_mu, z_log_std = bn_out.chunk(2, -1)
        z_dist = D.Normal(z_mu, F.softplus(z_log_std))
        if self.training:
            z = z_dist.rsample()
        else:
            z = z_dist.mean
        kl_loss = D.kl.kl_divergence(z_dist, D.Normal(0, 1)).mean()
        if self.tanh_latents:
            z = th.tanh(z)

        z = z.view(B, T, -1).permute(0, 2, 1).contiguous()  # [BT,C] -> BCT
        x_dec = self.decoder(z)
        x_out = self.postprocess(x_dec)

        extra = {
            'zdist': bn_out.view(B, T, -1).permute(0, 2, 1),
            'z': z,
            'out': x_out,
            'kl': kl_loss,
            'entropy': z_dist.entropy().mean(),
        }
        return x_out, extra
