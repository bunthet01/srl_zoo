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
    from ..preprocessing.utils import one_hot, gaussian_target
    import sys
    sys.path.append("..")
    from real_robots.constants import MIN_X, MAX_X, MIN_Y, MAX_Y
except:
    # absolute import: when executing directly: python train.py ...
    from models.base_models import BaseModelVAE
    from losses.losses import kullbackLeiblerLoss, generationLoss, KLDloss, BCEloss
    from preprocessing.utils import one_hot, gaussian_target
    import sys
    sys.path.append("..")
    from real_robots.constants import MIN_X, MAX_X, MIN_Y, MAX_Y


class DenseCVAE(BaseModelVAE):
    """
    Dense CVAE network
    :param state_dim: (int)
    :param img_shape: (tuple)
    """

    def __init__(self, state_dim, class_dim, img_shape,device):
        super(DenseCVAE, self).__init__(state_dim=state_dim, img_shape=img_shape)

        self.img_shape = img_shape
        self.class_dim = class_dim
        self.state_dim = state_dim
        self.device = device

        self.encoder_fc1 = nn.Linear(np.prod(self.img_shape)+class_dim+2, 50)
        self.encoder_fc21 = nn.Linear(50, state_dim)
        self.encoder_fc22 = nn.Linear(50, state_dim)

        self.decoder = nn.Sequential(
            nn.Linear(state_dim+self.class_dim+2, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, np.prod(self.img_shape)),
            nn.Tanh()
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode_cvae(self, x, c, t):
        # Flatten input
        x = x.view(x.size(0), -1)
        concat_input = th.cat([x, one_hot(c).to(self.device), t.float()], 1)
        concat_input = self.relu(self.encoder_fc1(concat_input))
        return self.encoder_fc21(concat_input), self.encoder_fc22(concat_input)

    def decode_cvae(self, z, c, t):
        concat_input = th.cat([z, one_hot(c).to(self.device), t.float()], 1)
        return self.decoder(concat_input).view(-1, self.img_shape[0], self.img_shape[1], self.img_shape[2] )
    
    def compute_tensor_cvae(self, x, c, t):
        """
        :param x: (th.Tensor)
        :param c: (th.Tensor)
        :return: (th.Tensor)
        """
        input_shape = x.size()
        mu, logvar = self.encode_cvae(x, c, t)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode_cvae(z, c, t).view(input_shape)
        return decoded, mu, logvar  
         


class CNNCVAE(BaseModelVAE):
    """
    Convolutional neural network for Conditional variational auto-encoder

    """
    def __init__(self, state_dim=3,class_dim=1, img_shape=(3, 224, 224),device='cpu'):
        super(CNNCVAE, self).__init__(state_dim=state_dim, img_shape=img_shape)
        outshape = summary(self.encoder_conv, img_shape, show=False)  # [-1, channels, high, width]
        self.img_height, self.img_width = outshape[-2:]
        self.class_dim =  class_dim
        self.device = device
        self.encoder_fc1 = nn.Linear(self.img_height * self.img_width * 64+self.class_dim+2, state_dim)
        self.encoder_fc2 = nn.Linear(self.img_height * self.img_width * 64+self.class_dim+2, state_dim)
        self.decoder_fc = nn.Sequential(
            nn.Linear(state_dim+self.class_dim+2, self.img_height * self.img_width * 64)
            )
    def encode_cvae(self, x, c, t):
        """
        :param x: (th.Tensor)
        :param c: (th.Tensor)
        :return: (th.Tensor)

        """
        
        x = self.encoder_conv(x)
        x_vector = x.view(x.size(0), -1)
        concat_input = th.cat([x_vector, one_hot(c).to(self.device),t.float()], 1)
        return self.encoder_fc1(concat_input), self.encoder_fc2(concat_input)

    def decode_cvae(self, z, c, t):
        """
        :param z: (th.Tensor)
        :param c: (th.Tensor)
        :return: (th.Tensor)
        """
        concat_input = th.cat([z, one_hot(c).to(self.device), t.float()], 1)
        out_put_1 = self.decoder_fc(concat_input)
        out_put_2 = out_put_1.view(out_put_1.size(0), 64,self.img_height, self.img_width)
        return self.decoder_conv(out_put_2)

    def compute_tensor_cvae(self, x, c, t):
        """
        :param x: (th.Tensor)
        :param c: (th.Tensor)
        :return: (th.Tensor)
        """
        input_shape = x.size()
        mu, logvar = self.encode_cvae(x, c, t)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode_cvae(z, c, t).view(input_shape)
        return decoded, mu, logvar
        
class CNNCVAE_NEW(BaseModelVAE):
    """
    Convolutional neural network for Conditional variational auto-encoder

    """
    def __init__(self, state_dim=3,class_dim=1, img_shape=(3, 224, 224), device='cpu'):
        super(CNNCVAE_NEW, self).__init__(state_dim=state_dim, img_shape=img_shape)
        outshape = summary(self.encoder_conv, img_shape, show=False)  # [-1, channels, high, width]
        self.img_height, self.img_width = outshape[-2:]
        self.class_dim =  class_dim
        self.device = device
        self.encoder_fc1 = nn.Linear(self.img_height * self.img_width * 64, state_dim)
        self.encoder_fc2 = nn.Linear(self.img_height * self.img_width * 64, state_dim)
        self.decoder_fc = nn.Sequential(
            nn.Linear(state_dim+self.class_dim+2, self.img_height * self.img_width * 64)
            )
        #########
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
        self.one_hot = th.zeros(self.class_dim, self.class_dim)
        self.one_hot = self.one_hot.scatter(1, th.arange(self.class_dim).type(th.LongTensor).view(self.class_dim,1), 1).view(self.class_dim, self.class_dim, 1, 1)
        self.fill = th.zeros([self.class_dim, self.class_dim, self.img_shape[1], self.img_shape[2]])
        for i in range(self.class_dim):
            self.fill[i, i, :, :] = 1
            
        self.encoder_conv_new = nn.Sequential(
            # 224x224xN_CHANNELS -> 112x112x64
            nn.Conv2d(self.img_shape[0]+self.class_dim+1, 64, kernel_size=7, stride=2, padding=3, bias=False),
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
        
    def encode_cvae(self, x, c, t):
        """
        :param x: (th.Tensor)
        :param c: (th.Tensor)
        :return: (th.Tensor)

        """
        target_chanel = gaussian_target(self.img_shape, t, MAX_X, MIN_X, MAX_Y, MIN_Y).float().to(self.device)
        c = self.fill[c.long()].to(self.device)
        x = th.cat([x,c,target_chanel], 1)
        x = self.encoder_conv_new(x)
        x = x.view(x.size(0), -1)
        return self.encoder_fc1(x), self.encoder_fc2(x)

    def decode_cvae(self, z, c, t):
        """
        :param z: (th.Tensor)
        :param c: (th.Tensor)
        :return: (th.Tensor)
        """
        concat_input = th.cat([z, one_hot(c).to(self.device), t.float()], 1)
        out_put_1 = self.decoder_fc(concat_input)
        out_put_2 = out_put_1.view(out_put_1.size(0), 64,self.img_height, self.img_width)
        return self.decoder_conv(out_put_2)

    def compute_tensor_cvae(self, x, c, t):
        """
        :param x: (th.Tensor)
        :param c: (th.Tensor)
        :return: (th.Tensor)
        """
        input_shape = x.size()
        mu, logvar = self.encode_cvae(x, c, t)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode_cvae(z, c, t).view(input_shape)
        return decoded, mu, logvar


class CVAETrainer(nn.Module):
    def __init__(self, state_dim=2, class_dim=1, img_shape=(3, 224, 224),device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.img_shape = img_shape
        self.class_dim = class_dim
        self.device = device

    def build_model(self, model_type='custom_cnn'):
        assert model_type in ['custom_cnn','custom_cnn_2', 'linear', 'mlp'], 'The model must be one of [custom_cn, linear, mlp] '
        self.model_type = model_type
        if model_type == 'custom_cnn':
            self.model = CNNCVAE(self.state_dim,self.class_dim, self.img_shape, self.device)
        elif model_type == 'custom_cnn_2':
            self.model = CNNCVAE_NEW(self.state_dim,self.class_dim, self.img_shape, self.device)
        elif model_type == 'mlp':
            self.model = DenseCVAE(self.state_dim, self.class_dim, self.img_shape, self.device)
        else:
            raise NotImplementedError(
                "model type: ({}) not supported yet.".format(model_type))

    def train_on_batch(self, obs, next_obs,action, next_action,target_pos, next_target_pos, optimizer, loss_manager, valid_mode, device, beta, c):
        (decoded_obs, mu, logvar), (next_decoded_obs, next_mu, next_logvar) = self.model.compute_tensor_cvae(obs,action,target_pos), \
            self.model.compute_tensor_cvae(next_obs, next_action, next_target_pos)
        kullbackLeiblerLoss(mu, next_mu, logvar, next_logvar, loss_manager, beta, c)
        generationLoss(decoded_obs, next_decoded_obs, obs, next_obs, weight=1.0, loss_manager=loss_manager)
        if not valid_mode: 
            loss_manager.updateLossHistory()
        loss = loss_manager.computeTotalLoss()
        if not valid_mode:
            loss.backward()
            optimizer.step()
        else:
            pass
        loss = loss.item()
        return loss

    def reconstruct(self, x, c, t):
        return self.model.decode_cvae(self.model.encode_cvae(x,c,t)[0], c, t)

    def encode(self, x, c, t):
        return self.model.encode_cvae(x, c, t)

    def decode(self, z, c, t):
        return self.model.decode_cvae(z, c, t)

    def forward(self, x, c, t):
        return self.model.encode_cvae(x, c, t)[0] 

if __name__ == "__main__":
    print("Start")

    img_shape = (3, 64, 64)
    model = CNNCVAE(state_dim=2, class_dim =1, img_shape=img_shape)
    A = summary(model, img_shape)
