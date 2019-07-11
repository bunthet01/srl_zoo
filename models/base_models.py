from __future__ import print_function, division, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils import spectral_norm
from torchsummary import summary


class BaseModelSRL(nn.Module):
    """
    Base Class for a SRL network
    It implements a getState method to retrieve a state from observations
    """

    def __init__(self, state_dim=2, img_shape=(3, 224, 224)):
        super(BaseModelSRL, self).__init__()
        # Do not define the attribute self.state_dim nor self.img_shape here.
        # e.g forward/inverse/reward models inherit also from BaseModelSRL

    def getStates(self, observations):
        """
        :param observations: (torch.Tensor)
        :return: (torch.Tensor)
        """
        return self.forward(observations)

    def forward(self, x):
        raise NotImplementedError


class BaseModelAutoEncoder(BaseModelSRL):
    """
    Base Class for a SRL network (autoencoder family)
    It implements a getState method to retrieve a state from observations
    """

    def __init__(self, state_dim=2, img_shape=(3, 224, 224)):
        super(BaseModelAutoEncoder, self).__init__(state_dim=state_dim, img_shape=img_shape)
        self.state_dim = state_dim
        self.img_shape = img_shape
        # Inspired by ResNet:
        # conv3x3 followed by BatchNorm2d
        self.encoder_conv = nn.Sequential(
            # 224x224xN_CHANNELS -> 112x112x64
            nn.Conv2d(self.img_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 56x56x64

            conv3x3(in_planes=64, out_planes=64, stride=1),  # 56x56x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27x64

            conv3x3(in_planes=64, out_planes=64, stride=2),  # 14x14x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 6x6x64
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 13x13x64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 27x27x64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 55x55x64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 111x111x64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, self.img_shape[0], kernel_size=4, stride=2),  # 224x224xN_CHANNELS
            nn.Tanh()
        )

    def getStates(self, observations):
        """
        :param observations: (torch.Tensor)
        :return: (torch.Tensor)
        """
        return self.encode(observations)

    def encode(self, x):
        """
        :param x: (torch.Tensor)
        :return: (torch.Tensor)
        """
        raise NotImplementedError

    def decode(self, x):
        """
        :param x: (torch.Tensor)
        :return: (torch.Tensor)
        """
        raise NotImplementedError

    def forward(self, x):
        """
        :param x: (torch.Tensor)
        :return: (torch.Tensor)
        """
        return self.encode(x)


class BaseModelVAE(BaseModelAutoEncoder):
    """
    Base Class for a SRL network (VAE family)
    It implements a getState method to retrieve a state from observations
    """

    def __init__(self, state_dim=2, img_shape=(3, 224, 224)):
        super(BaseModelVAE, self).__init__(state_dim=state_dim, img_shape=img_shape)

    # def getStates(self, observations):
    #     """
    #     :param observations: (torch.Tensor)
    #     :return: (torch.Tensor)
    #     """
    #     return self.encode(observations)[0]

    def encode(self, x):
        """
        :param x: (torch.Tensor)
        :return: (torch.Tensor)
        """
        raise NotImplementedError

    def decode(self, x):
        """
        :param x: (torch.Tensor)
        :return: (torch.Tensor)
        """
        raise NotImplementedError

    def reparameterize(self, mu, logvar):
        """
        Reparameterize for the backpropagation of z instead of q.
        (See "The reparameterization trick" section of https://arxiv.org/abs/1312.6114)
        :param mu: (torch.Tensor)
        :param logvar: (torch.Tensor)
        """
        if self.training:
            # logvar = \log(\sigma^2) = 2 * \log(\sigma)
            # \sigma = \exp(0.5 * logvar)
            std = logvar.mul(0.5).exp_()
            # Sample \epsilon from normal distribution
            # use std to create a new tensor, so we don't have to care
            # about running on GPU or not
            eps = std.new(std.size()).normal_()
            # Then multiply with the standard deviation and add the mean
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        """
        :param x: (torch.Tensor)
        :return: (torch.Tensor)
        """
        return self.encode(x)[0]

    def compute_tensors(self, x):
        input_shape = x.size()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z).view(input_shape)
        return decoded, mu, logvar


class CustomCNN(BaseModelSRL):
    """
    Convolutional Neural Network
    input shape : 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
    :param state_dim: (int)
    """

    def __init__(self, state_dim=2, img_shape=(3, 224, 224)):
        super(CustomCNN, self).__init__()
        self.state_dim = state_dim
        self.img_shape = img_shape
        # Inspired by ResNet:
        # conv3x3 followed by BatchNorm2d

        self.conv_layers = nn.Sequential(
            # 224x224x3 -> 112x112x64
            nn.Conv2d(self.img_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 56x56x64

            conv3x3(in_planes=64, out_planes=64, stride=1),  # 56x56x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27x64

            conv3x3(in_planes=64, out_planes=64, stride=2),  # 14x14x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 6x6x64
        )

        outshape = summary(self.conv_layers, img_shape, show=False)  # [-1, channels, high, width]
        self.img_height, self.img_width = outshape[-2:]
        self.fc = nn.Linear(self.img_height * self.img_width * 64, state_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """"
    From PyTorch Resnet implementation
    3x3 convolution with padding
    :param in_planes: (int)
    :param out_planes: (int)
    :param stride: (int)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def encodeOneHot(tensor, n_dim):
    """
    One hot encoding for a given tensor
    :param tensor: (torch Tensor)
    :param n_dim: (int) Number of dimensions
    :return: (torch.Tensor)
    """
    encoded_tensor = torch.Tensor(tensor.shape[0], n_dim).zero_().to(tensor.device)
    return encoded_tensor.scatter_(1, tensor.data, 1.)


class GaussianNoise(nn.Module):
    """
    Gaussian Noise layer
    :param batch_size: (int)
    :param input_dim: (int)
    :param std: (float) standard deviation
    :param mean: (float)
    :param device: (pytorch device)
    """

    def __init__(self, batch_size, input_dim, device, std, mean=0):
        super(GaussianNoise, self).__init__()
        self.std = std
        self.mean = mean
        self.device = device
        self.noise = torch.zeros(batch_size, input_dim, device=self.device)

    def forward(self, x):
        if self.training:
            self.noise.data.normal_(self.mean, std=self.std)
            return x + self.noise
        return x


class GaussianNoiseVariant(nn.Module):
    """
    Variant of the Gaussian Noise layer that does not require fixed batch_size
    It recreates a tensor at each call
    :param device: (pytorch device)
    :param std: (float) standard deviation
    :param mean: (float)
    """

    def __init__(self, device, std, mean=0):
        super(GaussianNoiseVariant, self).__init__()
        self.std = std
        self.mean = mean
        self.device = device

    def forward(self, x):
        if self.training:
            noise = torch.zeros(x.size(), device=self.device)
            noise.data.normal_(self.mean, std=self.std)
            return x + noise
        return x


def ConvSN2d(in_channels, out_channels, kernel_size,
             stride=1,
             padding=0,
             dilation=1,
             groups=1,
             bias=True,
             padding_mode='zeros'):
    A = spectral_norm(nn.Conv2d(in_channels, out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=groups,
                                bias=bias))
    # A.__class__.__name__ = 'ConvSN2d' ## [TODO]
    return A


def ConvTransposeSN2d(in_channels, out_channels, kernel_size,
                      stride=1,
                      padding=0,
                      output_padding=0,
                      groups=1,
                      bias=True,
                      dilation=1,
                      padding_mode='zeros'):
    A = spectral_norm(nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         output_padding=output_padding,
                                         groups=groups,
                                         bias=bias,
                                         dilation=dilation))
    # A.__class__.__name__ = 'ConvTransposeSN2d' ## [TODO]
    return A


def LinearSN(in_features, out_features, bias=True):
    A = spectral_norm(nn.Linear(in_features, out_features, bias=bias))
    # A.__class__.__name__ = 'LinearSN' ## [TODO]
    return A


class UNet(nn.Module):
    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        depth=4,
        start_ch=64,
        inc_rate=2,
        padding=True,
        batch_norm=True,
        spec_norm=False,
        dropout=0.5,
        up_mode='upconv',
        include_top=True
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            depth (int): depth of the network
            start_ch (int): number of filters in the first layer is start_ch
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            dropout (None or float): Use dropout (if not None) in Conv block.
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.include_top = include_top
        self.padding = padding
        self.out_ch = out_ch
        self.depth = depth
        self.spec_norm = spec_norm
        prev_channels = in_ch
        self.down_path = nn.ModuleList()
        for i in range(depth+1):
            self.down_path.append(
                UNetConvBlock(prev_channels, (inc_rate ** i) * start_ch,
                              padding, batch_norm, dropout, spec_norm=self.spec_norm)
            )
            prev_channels = (inc_rate ** i) * start_ch

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_path.append(
                UNetUpBlock(prev_channels, (inc_rate ** i) * start_ch, up_mode,
                            padding, batch_norm, dropout, spec_norm=self.spec_norm)
            )
            prev_channels = (inc_rate ** i) * start_ch

        if self.include_top:
            if self.spec_norm:
                self.last = ConvSN2d(prev_channels, out_ch, kernel_size=1)
            else:
                self.last = nn.Conv2d(prev_channels, out_ch, kernel_size=1)
        else:
            self.out_ch = prev_channels
        self.tanh_act = nn.Tanh()

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        if self.include_top:
            x = self.last(x)
            return self.tanh_act(x)
        else:
            return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, dropout, spec_norm=False):
        super(UNetConvBlock, self).__init__()
        self.spec_norm = spec_norm
        block = []
        if self.spec_norm:
            # [stride=1] padding = (k-1)/2
            block.append(ConvSN2d(in_size, out_size,
                                  kernel_size=3, padding=int(padding)))
        else:
            # [stride=1] padding = (k-1)/2
            block.append(nn.Conv2d(in_size, out_size,
                                   kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        if dropout is not None:
            block.append(nn.Dropout(p=dropout))
        if self.spec_norm:
            block.append(ConvSN2d(out_size, out_size,
                                  kernel_size=3, padding=int(padding)))
        else:
            block.append(nn.Conv2d(out_size, out_size,
                                   kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, dropout, spec_norm=False):
        super(UNetUpBlock, self).__init__()
        self.padding = padding
        self.spec_norm = spec_norm
        if up_mode == 'upconv':
            if self.spec_norm:
                self.up = ConvTransposeSN2d(
                    in_size, out_size, kernel_size=3, stride=2, padding=1, output_padding=1)
            else:
                # self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
                self.up = nn.ConvTranspose2d(
                    in_size, out_size, kernel_size=3, stride=2, padding=1, output_padding=1)
        elif up_mode == 'upsample':

            if self.spec_norm:
                self.up = nn.Sequential(
                    nn.Upsample(mode='bilinear', scale_factor=2),
                    ConvSN2d(in_size, out_size, kernel_size=1),
                )
            else:
                self.up = nn.Sequential(
                    nn.Upsample(mode='bilinear', scale_factor=2),
                    nn.Conv2d(in_size, out_size, kernel_size=1),
                )
        self.relu_act = nn.ReLU()
        self.conv_block = UNetConvBlock(
            in_size, out_size, padding, batch_norm, dropout, spec_norm=self.spec_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.relu_act(up)
        # print("Brige shape: {}, target size: {}".format(bridge.shape, up.shape[2:]))
        if self.padding:
            crop1 = bridge
        else:
            crop1 = self.center_crop(bridge, up.shape[2:])
        # print(up.shape)
        # print(crop1.shape)
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


if __name__ == "__main__":
    print("Start")

    img_shape = (3, 128, 128)
    model = CustomCNN(state_dim=2, img_shape=img_shape)
    A = summary(model, img_shape)
