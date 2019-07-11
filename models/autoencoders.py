from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F  # HACK TODO DEBUG
import numpy as np
from torchsummary import summary
try:
    # relative import: when executing as a package: python -m ...
    from .base_models import BaseModelAutoEncoder, ConvSN2d, ConvTransposeSN2d, LinearSN, UNet
    from ..losses.losses import autoEncoderLoss, AEboundLoss
    from .base_trainer import BaseTrainer
    from .gan import GeneratorUnet, EncoderUnet
except:
    # absolute import: when executing directly: python train.py ...
    from models.base_models import BaseModelAutoEncoder, ConvSN2d, ConvTransposeSN2d, LinearSN, UNet
    from losses.losses import autoEncoderLoss, AEboundLoss
    from models.base_trainer import BaseTrainer
    from models.gan import GeneratorUnet, EncoderUnet


class LinearAutoEncoder(BaseModelAutoEncoder):
    """
    :param state_dim: (int)
    :param img_shape: (tuple)
    """

    def __init__(self, state_dim, img_shape):
        super(LinearAutoEncoder, self).__init__(state_dim, img_shape)
        self.img_shape = img_shape

        self.encoder = nn.Sequential(
            nn.Linear(np.prod(self.img_shape), state_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(state_dim, np.prod(self.img_shape)),
            nn.Tanh()
        )

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        # Flatten input
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        x = self.decoder(x)
        x = x.view(x.size(0), *self.img_shape)
        return x


class DenseAutoEncoder(BaseModelAutoEncoder):
    """
    Dense autoencoder network
    Known issue: it reconstructs the image but omits the robot arm
    :param state_dim: (int)
    :param img_shape: (tuple)
    """

    def __init__(self, state_dim, img_shape):
        super(DenseAutoEncoder, self).__init__(state_dim, img_shape)
        self.img_shape = img_shape
        self.encoder = nn.Sequential(
            nn.Linear(np.prod(self.img_shape), 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, state_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(state_dim, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, np.prod(self.img_shape)),
            nn.Tanh()
        )

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        # Flatten input
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        x = self.decoder(x)
        x = x.view(x.size(0), *self.img_shape)
        return x


class CNNAutoEncoder(BaseModelAutoEncoder):
    """
    Custom convolutional autoencoder network
    Input dim (same as ResNet): 3x224x224
    :param state_dim: (int)
    """

    def __init__(self, state_dim=3, img_shape=(3, 224, 224)):
        # state_dim=state_dim, img_shape=img_shape
        super(CNNAutoEncoder, self).__init__(state_dim=state_dim, img_shape=img_shape)
        self.state_dim = state_dim
        self.img_shape = img_shape

        outshape = summary(self.encoder_conv, img_shape, show=False)  # [-1, channels, high, width]
        self.img_height, self.img_width = outshape[-2:]

        self.encoder_fc = nn.Sequential(
            nn.Linear(self.img_height * self.img_width * 64, state_dim)
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(state_dim, self.img_height * self.img_width * 64)
        )

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        encoded = self.encoder_conv(x)
        encoded = encoded.view(encoded.size(0), -1)
        return self.encoder_fc(encoded)

    def decode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        decoded = self.decoder_fc(x)
        decoded = decoded.view(x.size(0), 64, self.img_height, self.img_width)
        return self.decoder_conv(decoded)


class UNetEncoder(nn.Module):
    """

    Note: Only Encoder has getStates method.
    """

    def __init__(self, img_shape, state_dim,
                 unet_depth=2,  # 3
                 unet_ch=16,
                 unet_bn=False,
                 unet_drop=0.0,
                 spectral_norm=False):
        super().__init__()
        assert img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        self.state_dim = state_dim
        self.img_shape = img_shape
        self.spectral_norm = spectral_norm
        self.unet_depth = unet_depth
        self.unet_ch = unet_ch
        self.unet_drop = unet_drop
        self.unet_bn = unet_bn
        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU(negative_slope=0.2)],
            ['prelu', nn.PReLU()],
            ['tanh', nn.Tanh()],
            ['relu', nn.ReLU()],
            ['sigmoid', nn.Sigmoid()]
        ])
        self.unet = UNet(in_ch=self.img_shape[0], include_top=False, depth=self.unet_depth, start_ch=self.unet_ch,
                         batch_norm=self.unet_bn, spec_norm=self.spectral_norm, dropout=self.unet_drop, up_mode='upconv', out_ch=1)
        prev_channels = self.unet.out_ch

        self.modules_list = nn.ModuleList([])

        if self.spectral_norm:
            inter_features = 1 * (self.img_shape[1]//2**2) * (self.img_shape[2]//2**2)
            self.modules_list.append(ConvSN2d(prev_channels, 1, kernel_size=4, stride=2, padding=1))
            self.modules_list.append(self.activations['lrelu'])
            self.modules_list.append(ConvSN2d(1, 1, kernel_size=4, stride=2, padding=1))
            self.modules_list.append(self.activations['lrelu'])
            self.before_last = LinearSN(inter_features, 100, bias=True)
            self.last = LinearSN(100, self.state_dim, bias=True)
        else:
            inter_features = 1 * (self.img_shape[1]//2**2) * (self.img_shape[2]//2**2)
            self.modules_list.append(nn.Conv2d(prev_channels, 1, kernel_size=4, stride=2, padding=1))
            self.modules_list.append(self.activations['lrelu'])
            self.modules_list.append(nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1))
            self.modules_list.append(self.activations['lrelu'])
            self.before_last = nn.Linear(inter_features, 100, bias=True)
            self.last = nn.Linear(100, self.state_dim, bias=True)

    def forward(self, x):
        x = self.unet(x)
        for layer in self.modules_list:
            x = layer(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.before_last(x)
        x = self.activations['lrelu'](x)
        x = self.last(x)
        return x


class UNetGenerator(nn.Module):
    def __init__(self, img_shape, state_dim,
                 unet_depth=2,  # 3
                 unet_ch=16,  # 32
                 spectral_norm=False,
                 unet_bn=False,
                 unet_drop=0.0):
        super().__init__()
        self.state_dim = state_dim
        self.img_shape = img_shape
        self.spectral_norm = spectral_norm
        self.unet_depth = unet_depth
        self.unet_ch = unet_ch
        self.unet_drop = unet_drop
        self.unet_bn = unet_bn
        # self.lipschitz_G = 1.1 [TODO]
        assert self.img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        if self.spectral_norm:
            # state_layer = DenseSN(np.prod(self.img_shape), activation=None, lipschitz=self.lipschitz_G)(state_input)
            self.first = LinearSN(
                self.state_dim, np.prod(self.img_shape), bias=True)
        else:
            self.first = nn.Linear(
                self.state_dim, np.prod(self.img_shape), bias=True)

        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU(negative_slope=0.2)],
            ['prelu', nn.PReLU()],
            ['tanh', nn.Tanh()],
            ['relu', nn.ReLU()]
        ])

        out_channels = self.img_shape[0]  # = 3
        in_channels = out_channels
        self.unet = UNet(in_ch=in_channels, include_top=False, depth=self.unet_depth, start_ch=self.unet_ch,
                         batch_norm=self.unet_bn, spec_norm=self.spectral_norm, dropout=self.unet_drop, up_mode='upconv', out_ch=out_channels)
        prev_channels = self.unet.out_ch
        if self.spectral_norm:
            self.last = ConvSN2d(prev_channels, out_channels,
                                 kernel_size=3, stride=1, padding=1)
        else:
            self.last = nn.Conv2d(
                prev_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.first(x)
        x = self.activations['lrelu'](x)
        x = x.view(x.size(0), *self.img_shape)
        x = self.unet(x)
        x = self.last(x)
        x = self.activations['tanh'](x)
        return x


class UNetAutoEncoder(BaseModelAutoEncoder):
    """
    Custom UNet autoencoder network
    :param state_dim: (int)
    """

    def __init__(self, state_dim=3, img_shape=(3, 224, 224)):
        super(UNetAutoEncoder, self).__init__(state_dim=state_dim, img_shape=img_shape)
        self.decoder = GeneratorUnet(state_dim, img_shape, unet_bn=True)
        self.encoder = EncoderUnet(state_dim, img_shape, unet_bn=True)

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.encoder(x)

    def decode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.decoder(x)


class AutoEncoderTrainer(BaseTrainer):
    def __init__(self, state_dim=2, img_shape=(3, 224, 224)):
        super().__init__()
        # BaseTrainer.__init__(self)
        self.state_dim = state_dim
        self.img_shape = img_shape

    def build_model(self, model_type='custom_cnn'):
        assert model_type in ['custom_cnn', 'linear', 'mlp', 'unet']
        if model_type == 'custom_cnn':
            self.model = CNNAutoEncoder(self.state_dim, self.img_shape)
        elif model_type == 'mlp':
            self.model = DenseAutoEncoder(self.state_dim, self.img_shape)
        elif model_type == 'linear':
            self.model = LinearAutoEncoder(self.state_dim, self.img_shape)
        elif model_type == 'unet':
            self.model = UNetAutoEncoder(self.state_dim, self.img_shape)
        else:
            raise NotImplementedError("model type: ({}) not supported yet.".format(model_type))

    def train_on_batch(self, obs, next_obs, optimizer, loss_manager, valid_mode=False, device=torch.device('cpu')):
        state_pred = self.encode(obs)
        decoded_obs = self.decode(state_pred)
        decoded_next_obs = self.reconstruct(next_obs)
        autoEncoderLoss(obs, decoded_obs, next_obs, decoded_next_obs, weight=1.0, loss_manager=loss_manager)
        AEboundLoss(state_pred, weight=1.0, loss_manager=loss_manager, name='bonud_state_loss', max_val=50)
        loss = self.update_nn_weights(optimizer, loss_manager, valid_mode=valid_mode)
        return loss

    def reconstruct(self, x):
        return self.model.decode(self.model.encode(x))

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, x):
        return self.model.decode(x)

    def forward(self, x):
        return self.model.encode(x)  # or self.model(x)


if __name__ == "__main__":
    print("Start")
    from torchsummary import summary
    img_shape = (3, 128, 128)
    model = CNNAutoEncoder(state_dim=2, img_shape=img_shape)
    A = summary(model, img_shape)
    # import ipdb; ipdb.set_trace()
