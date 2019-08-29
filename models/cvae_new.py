# cvae.py is designed to use: "action" or ("action", "target position") as condition. cvae_new.py is designed to use ("target position" and "robot position") as condition 

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
    from real_robots.constants import MIN_X, MAX_X, MIN_Y, MAX_Y	# working with "Omnirobot-env"
except:
    # absolute import: when executing directly: python train.py ...
    from models.base_models import BaseModelVAE
    from losses.losses import kullbackLeiblerLoss, generationLoss, KLDloss, BCEloss
    from preprocessing.utils import one_hot, gaussian_target
    import sys
    sys.path.append("..")
    from real_robots.constants import MIN_X, MAX_X, MIN_Y, MAX_Y	# working with "Omnirobot-env"

        
class CNNCVAE_NEW(BaseModelVAE):
    """
    Convolutional neural network for Conditional variational auto-encoder.
    Two conditions are used: robot_position and target_position.

    """
    def __init__(self, state_dim=3, img_shape=(3, 224, 224), device='cpu'):
        super(CNNCVAE_NEW, self).__init__(state_dim=state_dim, img_shape=img_shape)
        outshape = summary(self.encoder_conv, img_shape, show=False)  # [-1, channels, high, width]
        self.img_height, self.img_width = outshape[-2:]
        self.device = device
        self.encoder_fc1 = nn.Linear(self.img_height * self.img_width * 64, state_dim)
        self.encoder_fc2 = nn.Linear(self.img_height * self.img_width * 64, state_dim)
        self.decoder_fc = nn.Sequential(
            nn.Linear(state_dim+4, self.img_height * self.img_width * 64)
            )
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
        self.encoder_conv_new = nn.Sequential(
            nn.Conv2d(self.img_shape[0]+2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  

            conv3x3(in_planes=64, out_planes=64, stride=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  

            conv3x3(in_planes=64, out_planes=64, stride=2),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  
        )
        
    def encode_cvae(self, x, t, r ):
        """
        :param x: (th.Tensor)
        :param t: (th.Tensor) target_position
        :param r: (th.Tensor) robot_position 
        :return: (th.Tensor)

        """
        
        target_chanel = gaussian_target(self.img_shape, t, MAX_X, MIN_X, MAX_Y, MIN_Y).float().to(self.device)
        robot_chanel = gaussian_target(self.img_shape, r, MAX_X, MIN_X, MAX_Y, MIN_Y).float().to(self.device) # gaussian_target is the same as gaussian_robot
        x = th.cat([x,target_chanel, robot_chanel], 1)
        x = self.encoder_conv_new(x)
        x = x.view(x.size(0), -1)
        return self.encoder_fc1(x), self.encoder_fc2(x)

    def decode_cvae(self, z, t, r):
        """
        :param z: (th.Tensor)
        :param t: (th.Tensor) target_position
        :param r: (th.Tensor) robot_position 
        :return: (th.Tensor)

        """
        concat_input = th.cat([z, t.float(), r.float()], 1)
        out_put_1 = self.decoder_fc(concat_input)
        out_put_2 = out_put_1.view(out_put_1.size(0), 64,self.img_height, self.img_width)
        return self.decoder_conv(out_put_2)

    def compute_tensor_cvae(self, x, t, r):
        """
        :param x: (th.Tensor)
        :param t: (th.Tensor) target_position
        :param r: (th.Tensor) robot_position 
        :return: (th.Tensor)

        """
        input_shape = x.size()
        mu, logvar = self.encode_cvae(x, t, r)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode_cvae(z, t, r).view(input_shape)
        return decoded, mu, logvar


class CVAE_NEW_Trainer(nn.Module):
    def __init__(self, state_dim=2, img_shape=(3, 224, 224),device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.img_shape = img_shape
        self.device = device

    def build_model(self, model_type='custom_cnn_2'):
        assert model_type in ['custom_cnn_2'], 'The model must be custom_cnn_2 '
        # [TODO] add other models 
        self.model_type = model_type
        if model_type == 'custom_cnn_2':
            self.model = CNNCVAE_NEW(self.state_dim, self.img_shape, self.device)
        else:
            raise NotImplementedError(
                "model type: ({}) not supported yet.".format(model_type))

    def train_on_batch(self, obs, next_obs,target_pos, next_target_pos,robot_pos, next_robot_pos, optimizer, loss_manager, valid_mode, device, beta, c):
        (decoded_obs, mu, logvar), (next_decoded_obs, next_mu, next_logvar) = self.model.compute_tensor_cvae(obs,target_pos,robot_pos ), \
            self.model.compute_tensor_cvae(next_obs, next_target_pos, next_robot_pos)
        kullbackLeiblerLoss(mu, next_mu, logvar, next_logvar, loss_manager, beta, c)
        generationLoss(decoded_obs, next_decoded_obs, obs, next_obs, weight=100.0, loss_manager=loss_manager) 
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

    def reconstruct(self, x, tar_pos, robot_pos):
        return self.model.decode_cvae(self.model.encode_cvae(x, tar_pos, robot_pos)[0], tar_pos, robot_pos)

    def encode(self, x, tar_pos, robot_pos):
        return self.model.encode_cvae(x, tar_pos, robot_pos)

    def decode(self, z, tar_pos, robot_pos):
        return self.model.decode_cvae(z, tar_pos, robot_pos)

    def forward(self, x, tar_pos, robot_pos):
        return self.model.encode_cvae(x, tar_pos, robot_pos)[0] 

if __name__ == "__main__":
    print("Start")

    img_shape = (3, 64, 64)
    model = CNNCVAE(state_dim=2, img_shape=img_shape)
    A = summary(model, img_shape)
