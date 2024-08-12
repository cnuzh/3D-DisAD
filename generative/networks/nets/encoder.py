import torch
import torch.nn as nn
from monai.networks.blocks import ResidualUnit, Convolution
from monai.networks.nets import resnet


class Encoder3D(nn.Module):
    def __init__(self, in_channels=1, input_size=64, dims=128, bn=True, latent_dim=192):
        super(Encoder3D, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            Convolution(3, in_channels, dims, kernel_size=4, strides=2, padding=1,
                        adn_ordering='DNA', act='RELU', norm='BATCH' if bn else None),
            # [B, dims, input_size/2, input_size/2, input_size/2]
            Convolution(3, dims, dims, kernel_size=4, strides=2, padding=1,
                        adn_ordering='DNA', act='RELU', norm='BATCH' if bn else None),
            # [B, dims, input_size/4, input_size/4, input_size/4]
            Convolution(3, dims, dims, kernel_size=4, strides=2, padding=1,
                        adn_ordering='DNA', act=None, norm='BATCH' if bn else None),
            # [B, dims, input_size/8, input_size/8, input_size/8]
            Convolution(3, dims, dims, kernel_size=4, strides=2, padding=1,
                        adn_ordering='DNA', act=None, norm='BATCH' if bn else None),
            # [B, dims, input_size/16, input_size/16, input_size/16]
            Convolution(3, dims, dims, kernel_size=4, strides=2, padding=1,
                        adn_ordering='DNA', act='RELU', norm='BATCH' if bn else None),
            # [B, dims, input_size/32, input_size/32, input_size/32]
            ResidualUnit(3, dims, dims, act='RELU', norm="BATCH" if bn else None),
            # [B, dims, input_size/32, input_size/32, input_size/32]
            ResidualUnit(3, dims, dims, act='RELU', norm="BATCH" if bn else None),
            # [B, dims, input_size/32, input_size/32, input_size/32]
            nn.Flatten(),
            # [B, dims * ((input_size / 32) ** 3)]
            nn.Linear(int(dims * ((input_size / 32) ** 3)), self.latent_dim),
            # [B, latent_dim]
        )

    def forward(self, x):
        return self.encoder(x)

# class Encoder3D(nn.Module):
#     def __init__(self, in_channels=1, input_size=64, dims=128, bn=True, latent_dim=192):
#         super(Encoder3D, self).__init__()
#         self.latent_dim = latent_dim
#         self.encoder = resnet.ResNet(
#             block=resnet.ResNetBlock, layers=[2, 2, 2, 2], block_inplanes=resnet.get_inplanes(),
#             spatial_dims=3, n_input_channels=in_channels, num_classes=self.latent_dim
#         )
#
#     def forward(self, x):
#         return self.encoder(x)


if __name__ == '__main__':
    model = Encoder3D(in_channels=1, dims=64, bn=True, latent_dim=128)
    print(model)
    inputs = torch.randn((5, 1, 64, 64, 64))
    outputs = model(inputs)
    print(outputs.shape)
