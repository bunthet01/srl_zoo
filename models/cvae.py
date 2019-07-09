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

class CNNCVAE(BaseModelVAE):
	"""
	Convolutional neural network for Conditional variational auto-encoder

	"""
    def __init__(self, state_dim=3,class_dim=1, img_shape=(3, 224, 224)):
        super(CNNVAE, self).__init__(state_dim=state_dim, img_shape=img_shape)
        outshape = summary(self.encoder_conv, img_shape, show=False)  # [-1, channels, high, width]
        self.img_height, self.img_width = outshape[-2:]
        self.class_dim =  class_dim

        self.encoder_fc1 = nn.Linear(self.img_height * self.img_width * 64+self.class_dim, state_dim)
        self.encoder_fc2 = nn.Linear(self.img_height * self.img_width * 64+self.class_dim, state_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(state_dim+self.class_dim, self.img_height * self.img_width * 64)
        )

    def encode_cvae(self, x, c):
        """
        :param x: (th.Tensor)
        :param c: (th.Tensor)
        :return: (th.Tensor)
        """
        x = self.encoder_conv(x)
        x_vector = x.view(x.size(0), -1)
        concat_input = th.cat([x_vector, c], 1)
        return self.encoder_f1(concat_input), self.encoder_f2(concat_input)

    def decode_cvae(self, z, c):
        """
        :param z: (th.Tensor)
        :param c: (th.Tensor)
        :return: (th.Tensor)
        """
        concat_input = th.cat([z, c], 1)
        out_put_1 = self.decode_fc(concat_input)
        out_put_2 = out_put_1.view(out_put_1.size(0), 64, 6, 6)
        return self.decoder_conv(out_put_2)

    def compute_tensor_cvae(self, x, c):
        """
        :param x: (th.Tensor)
        :param c: (th.Tensor)
        :return: (th.Tensor)
        """
        input_shape = x.size()
        mu, logvar = self.encode_cvae(x, c)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode_cvae(z, c).view(input_shape)
        return decoded, mu, logvar

    # def getStates_cvae(self, x, c):
    #     """
    #     :param x: (th.Tensor) observation
    #     :param c: (th.Tensor)
    #     :return: (th.Tensor)
    #     """
    #     mu, logvar = self.encode_cvae(x, c)
    #     return self.reparameterize(mu, logvar)


class CVAETrainer(nn.Module):
    def __init__(self, state_dim=2, class_dim=1, img_shape=(3, 224, 224)):
        super().__init__()
        self.state_dim = state_dim
        self.img_shape = img_shape
        self.class_dim = class_dim

    def build_model(self, model_type='custom_cnn'):
        assert model_type in ['custom_cnn', 'linear', 'mlp']
        srl_model = CNNCVAE(self.state_dim,self.class_dim, self.img_shape)

    def train_on_batch(self, obs, next_obs,action, next_action, optimizer, loss_manager, valid_mode=False, device=torch.device('cpu'), beta=1.0):
        (decoded_obs, mu, logvar), (next_decoded_obs, next_mu, next_logvar) = self.model.compute_tensor_cvae(obs,action), \
            self.model.compute_tensor_cvae(next_obs, next_action)
        # states, next_states = self.model(obs), self.model(next_obs)
        kullbackLeiblerLoss(mu, next_mu, logvar, next_logvar, loss_manager=loss_manager, beta=beta)
        generationLoss(decoded_obs, next_decoded_obs, obs, next_obs, weight=0.5e-6, loss_manager=loss_manager)
        loss_manager.updateLossHistory()
        loss = loss_manager.computeTotalLoss()
        if not valid_mode:
            loss.backward()
            optimizer.step()
        else:
            pass
        loss = loss.item()
        return loss

    def reconstruct(self, x, c):
        return self.model.decode_cvae(self.model.encode(x)[0], c)

    def encode(self, x, c):
        return self.model.encode_cvae(x, c)

    def decode(self, x):
        return self.model.decode_cvae(x, c)

    def forward(self, x):
        return self.model.encode_cvae(x, c)[0]  # or self.model(x)


if __name__ == "__main__":
    print("Start")

    img_shape = (3, 128, 128)
    model = CNNCVAE(state_dim=2, class_dim =1, img_shape=img_shape)
    A = summary(model, img_shape)