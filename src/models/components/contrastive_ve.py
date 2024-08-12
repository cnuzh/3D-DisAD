import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence
from monai.networks.blocks import Convolution
from itertools import chain


class ContrastiveVE(torch.nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int = 1,
            feature_size: int = 64,
            num_channels: Sequence[int] = (64, 128),
            latent_channels: int = 32,
    ):
        super(ContrastiveVE, self).__init__()

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.feature_size = feature_size
        self.num_channels = num_channels
        self.latent_channels = latent_channels

        blocks = list()
        output_channel = self.in_channels
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]

            blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=3,
                    strides=2,
                    padding=1,
                    act='relu',
                    dropout=None,
                    norm=None,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.flatten = nn.Flatten()

        self.z_mu_fc = nn.Linear(
            int(self.num_channels[-1] * (self.feature_size / (2 ** len(self.num_channels))) ** spatial_dims),
            128
        )
        self.z_mu = nn.Linear(
            128,
            self.latent_channels
        )

        self.z_var_fc = nn.Linear(
            int(self.num_channels[-1] * (self.feature_size / (2 ** len(self.num_channels))) ** spatial_dims),
            128
        )
        self.z_var = nn.Linear(
            128,
            self.latent_channels
        )

    def sample(self, z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z_vae = z_mu + eps * std
        return z_vae

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        for block in self.blocks:
            h = block(h)
        h = self.flatten(h)

        z_mu = self.z_mu_fc(h)
        z_mu = self.z_mu(z_mu)
        z_logvar = self.z_var_fc(h)
        z_logvar = self.z_var(z_logvar)

        return z_mu, z_logvar

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mu, z_logvar = self.encode(x)
        z = self.sample(z_mu, z_logvar)
        return z, z_mu, z_logvar
