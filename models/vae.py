from __future__ import print_function, division, absolute_import


import torch
import torch.nn as nn
from torchsummary import summary
try:
    # relative import: when executing as a package: python -m ...
    from .base_models import BaseModelVAE
    from ..losses.losses import kullbackLeiblerLoss, generationLoss
except:
    # absolute import: when executing directly: python train.py ...
    from models.base_models import BaseModelVAE
    from losses.losses import kullbackLeiblerLoss, generationLoss


class DenseVAE(BaseModelVAE):
    """
    Dense VAE network
    :param state_dim: (int)
    :param img_shape: (tuple)
    """

    def __init__(self, state_dim, img_shape):
        super(DenseVAE, self).__init__(state_dim=state_dim, img_shape=img_shape)

        self.img_shape = img_shape

        self.encoder_fc1 = nn.Linear(np.prod(self.img_shape), 50)
        self.encoder_fc21 = nn.Linear(50, state_dim)
        self.encoder_fc22 = nn.Linear(50, state_dim)

        self.decoder = nn.Sequential(
            nn.Linear(state_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, np.prod(self.img_shape)),
            nn.Tanh()
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        x = self.relu(self.encoder_fc1(x))
        return self.encoder_fc21(x), self.encoder_fc22(x)

    def decode(self, z):
        return self.decoder(z)


class CNNVAE(BaseModelVAE):
    """
    Custom convolutional VAE network
    Input dim (same as ResNet): 3x224x224
    :param state_dim: (int)
    """

    def __init__(self, state_dim=3, img_shape=(3, 224, 224)):
        super(CNNVAE, self).__init__(state_dim=state_dim, img_shape=img_shape)
        outshape = summary(self.encoder_conv, img_shape, show=False)  # [-1, channels, high, width]
        self.img_height, self.img_width = outshape[-2:]
        self.encoder_fc1 = nn.Linear(self.img_height * self.img_width * 64, state_dim)
        self.encoder_fc2 = nn.Linear(self.img_height * self.img_width * 64, state_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(state_dim, self.img_height * self.img_width * 64)
        )

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        return self.encoder_fc1(x), self.encoder_fc2(x)

    def decode(self, z):
        """
        :param z: (th.Tensor)
        :return: (th.Tensor)
        """
        z = self.decoder_fc(z)
        z = z.view(z.size(0), 64, self.img_height, self.img_width)
        return self.decoder_conv(z)


class VAETrainer(nn.Module):
    def __init__(self, state_dim=2, img_shape=(3, 224, 224)):
        super().__init__()
        self.state_dim = state_dim
        self.img_shape = img_shape

    def build_model(self, model_type='custom_cnn'):
        assert model_type in ['custom_cnn', 'linear', 'mlp']
        if model_type == 'custom_cnn':
            self.model = CNNVAE(self.state_dim, self.img_shape)
        elif model_type == 'mlp':
            self.model = DenseVAE(self.state_dim, self.img_shape)
        else:
            raise NotImplementedError(
                "model type: ({}) not supported yet.".format(model_type))

    def train_on_batch(self, obs, next_obs, optimizer, loss_manager, valid_mode=False, device=torch.device('cpu'), beta=1.0):
        (decoded_obs, mu, logvar), (next_decoded_obs, next_mu, next_logvar) = self.model.compute_tensors(obs), \
            self.model.compute_tensors(next_obs)
        # states, next_states = self.model(obs), self.model(next_obs)
        kullbackLeiblerLoss(mu, next_mu, logvar, next_logvar, loss_manager=loss_manager, beta=beta)
        generationLoss(decoded_obs, next_decoded_obs, obs, next_obs, weight=1.0, loss_manager=loss_manager) #0.5e-6
        loss_manager.updateLossHistory()
        loss = loss_manager.computeTotalLoss()
        if not valid_mode:
            loss.backward()
            optimizer.step()
        else:
            pass
        loss = loss.item()
        return loss

    def reconstruct(self, x):
        """
        This function is used for the visualization (testing/validation mode) only. Not for the training.
        """
        return self.model.decode(self.model.encode(x)[0])

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, x):
        return self.model.decode(x)

    def forward(self, x):
        """
        All srl model should return the 'state' via 'forward' function.
        """
        return self.model.encode(x)[0]  # or self.model(x)


if __name__ == "__main__":
    print("Start")

    img_shape = (3, 128, 128)
    model = CNNVAE(state_dim=2, img_shape=img_shape)
    A = summary(model, img_shape)
