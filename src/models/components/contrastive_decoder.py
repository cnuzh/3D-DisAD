import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence
from monai.networks.blocks import Convolution
from itertools import chain


class ContrastiveDecoder(torch.nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            out_channels: int = 1,
            feature_size: int = 64,
            num_channels: Sequence[int] = (64, 128),
            decode_num_channels: Sequence[int] = (32, 16),
            latent_channels: int = 32,
    ):
        super(ContrastiveDecoder, self).__init__()

        self.spatial_dims = spatial_dims
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.num_channels = num_channels
        self.decode_num_channels = decode_num_channels
        self.latent_channels = latent_channels

        self.decode_fc1 = nn.Linear(
            self.latent_channels,
            128
        )

        self.decode_fc2 = nn.Linear(
            128,
            int(self.num_channels[-1] * (self.feature_size / (2 ** len(self.num_channels))) ** spatial_dims)
        )

        decode_blocks = list()
        output_channel = self.num_channels[-1]
        for i in range(len(self.decode_num_channels)):
            input_channel = output_channel
            output_channel = decode_num_channels[i]

            decode_blocks.append(
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
                    is_transposed=True,
                )
            )
        self.decode_blocks = nn.ModuleList(decode_blocks)

        self.final_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.decode_num_channels[-1],
            out_channels=self.out_channels,
            kernel_size=3,
            strides=1,
            padding=1,
            act='sigmoid',
            dropout=None,
            norm=None,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.decode_fc1(z)
        z = self.decode_fc2(z)

        z = z.view(
            -1, self.num_channels[-1],
            *([int(self.feature_size / (2 ** len(self.num_channels)))] * self.spatial_dims)
        )

        for block in self.decode_blocks:
            z = block(z)

        z = self.final_block(z)

        return z
