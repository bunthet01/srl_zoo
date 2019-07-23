from __future__ import print_function, division, absolute_import

from torch.autograd import Variable
import torch as th
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
try:
    # relative import: when executing as a package: python -m ...
    from .base_models import BaseModelVAE
    from ..losses.losses import kullbackLeiblerLoss, generationLoss
except:
    # absolute import: when executing directly: python train.py ...
    from models.base_models import BaseModelVAE
    from losses.losses import kullbackLeiblerLoss, generationLoss, KLDloss, BCEloss

class DenseCVAE(BaseModelVAE):
    """
    Dense VAE network
    :param state_dim: (int)
    :param img_shape: (tuple)
    """

    def __init__(self, state_dim, class_dim, img_shape):
        super(DenseCVAE, self).__init__(state_dim=state_dim, img_shape=img_shape)

        self.img_shape = img_shape
        self.class_dim = class_dim
        self.state_dim = state_dim

        self.encoder_fc1 = nn.Linear(np.prod(self.img_shape)+class_dim, 50)
        self.encoder_fc21 = nn.Linear(50, state_dim)
        self.encoder_fc22 = nn.Linear(50, state_dim)

        self.decoder = nn.Sequential(
            nn.Linear(state_dim+self.class_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, np.prod(self.img_shape)),
            nn.Tanh()
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode_cvae(self, x, c):
        # Flatten input
        x = x.view(x.size(0), -1)
        concat_input = th.cat([x, c], 1)
        concat_input = self.relu(self.encoder_fc1(concat_input))
        return self.encoder_fc21(concat_input), self.encoder_fc22(concat_input)

    def decode_cvae(self, z, c):
        concat_input = th.cat([z, c], 1)
        return self.decoder(concat_input).view(-1, self.img_shape[0], self.img_shape[1], self.img_shape[2] )
    
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
         


class CNNCVAE(BaseModelVAE):
    """
    Convolutional neural network for Conditional variational auto-encoder

    """
    def __init__(self, state_dim=3,class_dim=1, img_shape=(3, 224, 224)):
        super(CNNCVAE, self).__init__(state_dim=state_dim, img_shape=img_shape)
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
        return self.encoder_fc1(concat_input), self.encoder_fc2(concat_input)

    def decode_cvae(self, z, c):
        """
        :param z: (th.Tensor)
        :param c: (th.Tensor)
        :return: (th.Tensor)
        """
        concat_input = th.cat([z, c], 1)
        out_put_1 = self.decoder_fc(concat_input)
        out_put_2 = out_put_1.view(out_put_1.size(0), 64,self.img_height, self.img_width)
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
        

class CVAETrainer(nn.Module):
    def __init__(self, state_dim=2, class_dim=1, img_shape=(3, 224, 224)):
        super().__init__()
        self.state_dim = state_dim
        self.img_shape = img_shape
        self.class_dim = class_dim

    def build_model(self, model_type='custom_cnn'):
        assert model_type in ['custom_cnn', 'linear', 'mlp'], 'The model must be one of [custom_cn, linear, mlp] '
        self.model_type = model_type
        if model_type == 'custom_cnn':
            self.model = CNNCVAE(self.state_dim,self.class_dim, self.img_shape)
        elif model_type == 'mlp':
            self.model = DenseCVAE(self.state_dim, self.class_dim, self.img_shape)
        else:
            raise NotImplementedError(
                "model type: ({}) not supported yet.".format(model_type))

    def train_on_batch(self, obs, next_obs,action, next_action, optimizer, loss_manager, valid_mode, device, beta, c):
        (decoded_obs, mu, logvar), (next_decoded_obs, next_mu, next_logvar) = self.model.compute_tensor_cvae(obs,action), \
            self.model.compute_tensor_cvae(next_obs, next_action)
        kullbackLeiblerLoss(mu, next_mu, logvar, next_logvar, loss_manager, beta, c)
        generationLoss(decoded_obs, next_decoded_obs, obs, next_obs, weight=1, loss_manager=loss_manager)
       
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
        return self.model.decode_cvae(self.model.encode_cvae(x,c)[0], c)

    def encode(self, x, c):
        return self.model.encode_cvae(x, c)

    def decode(self, z, c):
        return self.model.decode_cvae(z, c)

    def forward(self, x, c):
        return self.model.encode_cvae(x, c)[0] 

if __name__ == "__main__":
    print("Start")

    img_shape = (3, 128, 128)
    model = CNNCVAE(state_dim=2, class_dim =1, img_shape=img_shape)
    A = summary(model, img_shape)
