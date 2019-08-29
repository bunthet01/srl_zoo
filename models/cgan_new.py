# The difference between cgan.py and cgan_new.py is the training process. In cgan.py, we can add the encoder on top of gan to help the generator generate more realistic  images. And the training process is different. In cgan.py, the training step of generator and discriminator change depend the accuray of each neural network. The dataloader is pass to "train_on_batch" which slow down the training. In cgan_new.py, the training is done is the traditional way, so faster to train

import numpy as np
import torch
from torch import nn
import torchvision.utils as vutils
import torch.nn.functional as F
try:
    # relative import
    from .base_models import BaseModelSRL,BaseModelAutoEncoder, ConvSN2d, ConvTransposeSN2d, LinearSN, UNet
    from .base_trainer import BaseTrainer
    from ..losses.losses import ganNonSaturateLoss, autoEncoderLoss, ganBCEaccuracy, AEboundLoss
    from ..preprocessing.utils import one_hot, gaussian_target,sample_target_pos
    import sys
    sys.path.append("..")
    from real_robots.constants import MIN_X, MAX_X, MIN_Y, MAX_Y,TARGET_MAX_X, TARGET_MIN_X, TARGET_MAX_Y, TARGET_MIN_Y   # using Omnibot_env
except:
    from models.base_models import BaseModelSRL,BaseModelAutoEncoder, ConvSN2d, ConvTransposeSN2d, LinearSN, UNet
    from models.base_trainer import BaseTrainer
    from losses.losses import ganNonSaturateLoss, autoEncoderLoss, ganBCEaccuracy, AEboundLoss
    from preprocessing.utils import one_hot, gaussian_target,sample_target_pos
    import sys
    sys.path.append("..")
    from real_robots.constants import MIN_X, MAX_X, MIN_Y, MAX_Y,TARGET_MAX_X, TARGET_MIN_X, TARGET_MAX_Y, TARGET_MIN_Y   # using Omnibot_env
from torchsummary import summary

class GeneratorUnet(nn.Module):
    """
    GeneratorUnet. Two conditions are used : action and target_position.
    Parameter "only_action" is used to use only "action" as condition.
    :param label_dim: (th.Tensor) dimension of the action.
    """
    def __init__(self, state_dim, img_shape, label_dim,
                 unet_depth=2,  # 3
                 unet_ch=16,  # 32
                 spectral_norm=False, device='cpu', only_action=False,
                 unet_bn=False,
                 unet_drop=0.0):
        super().__init__()
        self.state_dim = state_dim
        self.img_shape = img_shape
        self.device = device
        self.only_action = only_action
        self.label_dim = label_dim
        self.spectral_norm = spectral_norm
        self.unet_depth = unet_depth
        self.unet_ch = unet_ch
        self.unet_drop = unet_drop
        self.unet_bn = unet_bn
        # self.lipschitz_G = 1.1 [TODO]
        assert self.img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        print("GeneratorUnet: sn = ", spectral_norm)
        if self.spectral_norm:
            # state_layer = DenseSN(np.prod(self.img_shape), activation=None, lipschitz=self.lipschitz_G)(state_input)
            if self.only_action:
                self.first = LinearSN(
                    self.state_dim + self.label_dim, np.prod(self.img_shape), bias=True)
            else: 
                self.first = LinearSN(
                    self.state_dim + self.label_dim+2, np.prod(self.img_shape), bias=True)  # dimension of target_position is 2
        else:
            if self.only_action:
                self.first = nn.Linear(
                    self.state_dim + self.label_dim, np.prod(self.img_shape), bias=True)
            else:
                self.first = nn.Linear(
                    self.state_dim + self.label_dim+2, np.prod(self.img_shape), bias=True)  # dimension of target_position is 2

            # state_layer = Dense(np.prod(self.img_shape), activation=None)(state_input)
        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU(negative_slope=0.2)],
            ['prelu', nn.PReLU()],
            ['tanh', nn.Tanh()],
            ['relu', nn.ReLU()]
        ])

        out_channels = self.img_shape[0]  # = 3
        in_channels = out_channels
        self.unet = UNet(in_ch=in_channels, include_top=False, depth=self.unet_depth, start_ch=self.unet_ch,
                         batch_norm=self.unet_bn, spec_norm=self.spectral_norm, dropout=self.unet_drop,
                         up_mode='upconv', out_ch=out_channels)
        prev_channels = self.unet.out_ch
        if self.spectral_norm:
            self.last = ConvSN2d(prev_channels, out_channels,
                                 kernel_size=3, stride=1, padding=1)
        else:
            self.last = nn.Conv2d(
                prev_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward_cgan(self, x, l, t, only_action):
        """
        :param x:(th.Tensor)
        :param l:(th.Tensor) label or action
        :param t:(th.Tensor) target position
        :param only_action: (boolean)
        
        """
        l = one_hot(l).to(self.device)
        if only_action:
            x = torch.cat([x, l], 1)
        else:
            x = torch.cat([x, l, t.float().to(self.device)], 1)
        x = self.first(x)
        x = self.activations['lrelu'](x)
        x = x.view(x.size(0), *self.img_shape)
        x = self.unet(x)
        x = self.last(x)
        x = self.activations['tanh'](x)
        return x


class GeneratorResNet(BaseModelAutoEncoder):
    """
    ResNet Generator. Two conditions are used : action and target_position.
    Parameter "only_action" is used to use only "action" as condition.
    :param label_dim: (th.Tensor) dimension of the action.
    """

    def __init__(self, state_dim, img_shape, label_dim, spectral_norm=False, device='cpu', only_action=False):
        super().__init__(state_dim=state_dim, img_shape=img_shape)
        assert img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        print("GeneratorResnet: sn = ", spectral_norm)
        self.state_dim = state_dim
        self.img_shape = img_shape
        self.device = device
        self.only_action = only_action
        self.label_dim = label_dim
        self.spectral_norm = spectral_norm

        outshape = summary(self.encoder_conv, self.img_shape, show=False) # [-1, channels, high, width] 
        self.in_height, self.in_width = outshape[-2:]
        if self.only_action:
            self.decoder_fc = nn.Sequential(
                nn.Linear(state_dim+label_dim, self.in_height*self.in_width*64))
        else:
            self.decoder_fc = nn.Sequential(
                nn.Linear(state_dim+label_dim+2, self.in_height*self.in_width*64))
        if self.spectral_norm:
            self.decoder_conv = nn.Sequential(
                ConvTransposeSN2d(64, 64, kernel_size=3, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                ConvTransposeSN2d(64, 64, kernel_size=3, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                ConvTransposeSN2d(64, 64, kernel_size=3, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                ConvTransposeSN2d(64, 64, kernel_size=3, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                ConvTransposeSN2d(64, self.img_shape[0], kernel_size=4, stride=2),
                nn.Tanh()
            )

        else:
            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.ConvTranspose2d(64, self.img_shape[0], kernel_size=4, stride=2),
                nn.Tanh()
            )
    def forward_cgan(self, x, l, t, only_action):
        """
        :param x:(th.Tensor)
        :param l:(th.Tensor) label or action
        :param t:(th.Tensor) target position
        :param only_action: (boolean)
        
        """
        if only_action: 
            decoded = torch.cat([x, one_hot(l).to(self.device)], 1)
        else:          
            decoded = torch.cat([x, one_hot(l).to(self.device), t.float().to(self.device)], 1)
        decoded = self.decoder_fc(decoded)
        decoded = decoded.view(x.size(0), 64, self.in_height, self.in_width)
        return self.decoder_conv(decoded)
    
        
class Discriminator(nn.Module):
        """
        Discriminator. Two conditions are used : action and target_position.
        Parameter "only_action" is used to use only "action" as condition.
        :param label_dim: (th.Tensor) dimension of the action.
        """
    def __init__(self, state_dim, img_shape,label_dim,
                 spectral_norm=False,device='cpu',only_action=False,
                 d_chs=16):  # 32
        super().__init__()
        self.img_shape = img_shape
        self.state_dim = state_dim
        self.label_dim = label_dim
        self.device = device
        self.only_action = only_action
        self.spectral_norm = spectral_norm
        self.d_chs = d_chs

        assert self.img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        print("Discriminator: sn = ", spectral_norm)
        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU(negative_slope=0.2)],
            ['prelu', nn.PReLU()],
            ['tanh', nn.Tanh()],
            ['relu', nn.ReLU()],
            ['sigmoid', nn.Sigmoid()]
        ])
        self.modules_list = nn.ModuleList([])
        COUNT_IMG_REDUCE = 0

        def d_layer(prev_channels, out_channels, kernel_size=4, spectral_norm=False):
            """Discriminator layer"""
            nonlocal COUNT_IMG_REDUCE
            COUNT_IMG_REDUCE += 1
            if spectral_norm:
                # [stride=2] padding = (kernel_size/2) -1
                layer = ConvSN2d(prev_channels, out_channels,
                                 kernel_size=kernel_size, stride=2, padding=1)
            else:
                # [stride=2] padding = (kernel_size/2) -1
                layer = nn.Conv2d(prev_channels, out_channels,
                                  kernel_size=kernel_size, stride=2, padding=1)
            return [layer, self.activations['lrelu']]  # , out.out_channels

        start_chs = self.img_shape[0]
        self.modules_list.extend(
            d_layer(start_chs, self.d_chs, spectral_norm=self.spectral_norm))
        self.modules_list.extend(
            d_layer(self.d_chs, self.d_chs*2, spectral_norm=self.spectral_norm))
        self.modules_list.extend(
            d_layer(self.d_chs*2, self.d_chs*4, spectral_norm=self.spectral_norm))
        self.modules_list.extend(
            d_layer(self.d_chs*4, self.d_chs*8, spectral_norm=self.spectral_norm))
        self.modules_list.extend(
            d_layer(self.d_chs*8, self.d_chs*8, spectral_norm=self.spectral_norm))

        if self.spectral_norm:
            self.modules_list.append(ConvSN2d(self.d_chs*8, self.d_chs*4,
                                              kernel_size=3, stride=1, padding=1))

            last_channels = self.modules_list[-1].out_channels   
            times = COUNT_IMG_REDUCE
            in_features = last_channels * \
                (self.img_shape[1]//2**times) * (self.img_shape[2]//2**times)  
            self.before_last = LinearSN(in_features, self.state_dim, bias=True)
            if self.only_action:
                self.last = LinearSN(self.state_dim+self.label_dim, 1, bias=True)
            else:
                self.last = LinearSN(self.state_dim+self.label_dim+2, 1, bias=True)
        else:
            self.modules_list.append(nn.Conv2d(self.d_chs*8, self.d_chs*4,
                                               kernel_size=3, stride=1, padding=1))
            last_channels = self.modules_list[-1].out_channels
            times = COUNT_IMG_REDUCE
            in_features = last_channels * \
                (self.img_shape[1]//2**times) * (self.img_shape[2]//2**times)
            self.before_last = nn.Linear(
                in_features, self.state_dim, bias=True)
            if self.only_action:
                self.last = nn.Linear(self.state_dim+self.label_dim, 1, bias=True)
            else:
                self.last = nn.Linear(self.state_dim+self.label_dim+2, 1, bias=True)

    def forward_cgan(self, x, l, t, only_action):
        """
        :param x:(th.Tensor)
        :param l:(th.Tensor) label or action
        :param t:(th.Tensor) target position
        :param only_action: (boolean)
        
        """
        for layer in self.modules_list:
            x = layer(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.activations['lrelu'](x)
        x = self.before_last(x)
        x = self.activations['lrelu'](x)
        if self.only_action:
            x = torch.cat([x,one_hot(l).to(self.device)],1)
        else:
            x = torch.cat([x,one_hot(l).to(self.device),t.float().to(self.device)],1)
        x = self.last(x)
        x = self.activations['sigmoid'](x)
        return x.squeeze()

# The DiscriminatorDC and GeneratorDC is inspired by https://github.com/pytorch/examples/tree/master/dcgan
class DiscriminatorDC(nn.Module):
    """
    Deep convolutional discriminator.
    Two conditions are used : action and target_position.
    Parameter "only_action" is used to use only "action" as condition.
    :param label_dim: (th.Tensor) dimension of the action.
    """
    def __init__(self, state_dim, label_dim, img_shape,spectral_norm=False,device='cpu',only_action=False):
        super(DiscriminatorDC, self).__init__()
        assert img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        print("DiscriminatorDC: sn = ", spectral_norm)
        self.device = device
        self.only_action = only_action
        self.img_shape = img_shape
        self.one_hot = torch.zeros(label_dim, label_dim)
        self.one_hot = self.one_hot.scatter(1, torch.arange(label_dim).type(torch.LongTensor).view(label_dim,1), 1).view(label_dim, label_dim, 1, 1)
        self.fill = torch.zeros([label_dim, label_dim, img_shape[1], img_shape[2]])
        for i in range(label_dim):
            self.fill[i, i, :, :] = 1
        ndf = 128
        assert ndf%4 == 0, 'This chanel number must be a multiple of 4'
        nc = int(img_shape[0])
        if spectral_norm:
            if self.only_action:
                self.conv1_1 = nn.Sequential(
                    ConvSN2d(nc, int(ndf/2), 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True))
                self.conv1_2 = nn.Sequential(
                    ConvSN2d(label_dim, int(ndf/2), 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                    )
            else:
                self.conv1_1 = nn.Sequential(
                    ConvSN2d(nc, int(ndf/2), 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True))
                self.conv1_2 = nn.Sequential(
                    ConvSN2d(label_dim, int(ndf/4), 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                    )
                self.conv1_3 = nn.Sequential(
                    ConvSN2d(1, int(ndf/4), 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                    )
            self.conv2 = nn.Sequential(
                ConvSN2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                ConvSN2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                ConvSN2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                ConvSN2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()


                )
        else:
            if self.only_action:
                self.conv1_1 = nn.Sequential(
                    nn.Conv2d(nc, int(ndf/2), 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True))
                self.conv1_2 = nn.Sequential(
                    nn.Conv2d(label_dim, int(ndf/2), 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                    )
            else:
                self.conv1_1 = nn.Sequential(
                    nn.Conv2d(nc, int(ndf/2), 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True))
                self.conv1_2 = nn.Sequential(
                    nn.Conv2d(label_dim, int(ndf/4), 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                    )
                self.conv1_3 = nn.Sequential(
                    nn.Conv2d(1, int(ndf/4), 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                    )
            self.conv2 = nn.Sequential(
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()


                )

    def forward_cgan(self, input, label, target, only_action):
        """
        :param input: (th.Tensor)
        :param label: (th.Tensor) label or action
        :param target: (th.Tensor) target position
        :param only_action: (boolean)
        
        """
        label = self.fill[label].to(self.device)
        target = gaussian_target(self.img_shape, target, MAX_X, MIN_X, MAX_Y, MIN_Y).float().to(self.device)
        x = self.conv1_1(input)
        y = self.conv1_2(label)
        if only_action:
            x = torch.cat([x,y],1)
        else:
            t = self.conv1_3(target)
            x = torch.cat([x,y,t],1)
        x = self.conv2(x)
        
        return x.view(-1, 1).squeeze(1)
        
class GeneratorDC(nn.Module):
    def __init__(self, state_dim, label_dim, img_shape, spectral_norm=False, device='cpu', only_action=False):
        super(GeneratorDC, self).__init__()
        assert img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        self.device = device 
        self.only_action = only_action
        self.img_shape = img_shape
        print("GeneratorDC: sn = ", spectral_norm)
        self.one_hot = torch.zeros(label_dim, label_dim)
        self.one_hot = self.one_hot.scatter(1, torch.arange(label_dim).type(torch.LongTensor).view(label_dim,1), 1).view(label_dim, label_dim, 1, 1)
        self.fill = torch.zeros([label_dim, label_dim, img_shape[1], img_shape[2]])
        for i in range(label_dim):
            self.fill[i, i, :, :] = 1
        nz = state_dim
        ngf = 128
        nc = img_shape[0]
        if spectral_norm:
            if self.only_action:
                self.deconv1_1 = nn.Sequential(
                    ConvTransposeSN2d(nz, ngf * 4, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True))
                self.deconv1_2 = nn.Sequential(
                    ConvTransposeSN2d(label_dim,ngf * 4, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True))
            else:
                self.deconv1_1 = nn.Sequential(
                    ConvTransposeSN2d(nz, ngf * 4, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True))
                self.deconv1_2 = nn.Sequential(
                    ConvTransposeSN2d(label_dim,ngf * 2, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True))
                self.deconv1_3 = nn.Sequential(
                    ConvTransposeSN2d(2,ngf * 2, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True))
            self.deconv2 = nn.Sequential(
                ConvTransposeSN2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),

                ConvTransposeSN2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),

                ConvTransposeSN2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                ConvTransposeSN2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                ) 
        else:
            if self.only_action:
                self.deconv1_1 = nn.Sequential(
                    nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True))
                self.deconv1_2 = nn.Sequential(
                    nn.ConvTranspose2d(label_dim,ngf * 4, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True))
            else:
                self.deconv1_1 = nn.Sequential(
                    nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True))
                self.deconv1_2 = nn.Sequential(
                    nn.ConvTranspose2d(label_dim,ngf * 2, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True))
                self.deconv1_3 = nn.Sequential(
                    nn.ConvTranspose2d(2,ngf * 2, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True))
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()

                )   
    def forward_cgan(self,input, label, target, only_action):
        """
        :param input: (th.Tensor)
        :param label: (th.Tensor) label or action
        :param target: (th.Tensor) target position
        :param only_action: (boolean)
        
        """
        input = input.view(input.size(0), input.size(1),1,1).to(self.device)
        label = self.one_hot[label].to(self.device)
        target = target[...,None][...,None].to(self.device)
        x = self.deconv1_1(input)
        y = self.deconv1_2(label)
        if only_action:
            x = torch.cat([x,y],1)
        else:
            t = self.deconv1_3(target)
            x = torch.cat([x,y,t],1)
        x = self.deconv2(x)
        return x      
        

class CGanNewTrainer(BaseTrainer):
    def __init__(self, state_dim, label_dim, img_shape,device,only_action):
        super().__init__()
        self.state_dim = state_dim
        self.img_shape = img_shape
        self.label_dim = label_dim
        self.device = device
        self.only_action = only_action
    
    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    
    def build_model(self, model_type='dc'):
        assert model_type in ['custom_cnn', 'unet', 'dc']
        if model_type == "dc":
            self.generator = GeneratorDC(self.state_dim, self.label_dim, self.img_shape, spectral_norm=True,device=self.device, only_action=self.only_action)
            self.discriminator = DiscriminatorDC(self.state_dim, self.label_dim, self.img_shape, spectral_norm=True,device=self.device, only_action=self.only_action)

            self.generator.apply(self.weights_init)
            self.discriminator.apply(self.weights_init)
        elif model_type == "unet":
            self.generator = GeneratorUnet(self.state_dim, self.img_shape, self.label_dim, spectral_norm=True,
                                           device=self.device, only_action=self.only_action)
            self.discriminator = Discriminator(self.state_dim, self.img_shape, self.label_dim, spectral_norm=True,
                                               device=self.device, only_action=self.only_action)
        elif model_type == "custom_cnn":
            self.generator = GeneratorResNet(self.state_dim, self.img_shape, self.label_dim,
                                             spectral_norm=True, device=self.device,only_action=self.only_action)
            self.discriminator = Discriminator(self.state_dim, self.img_shape, self.label_dim, spectral_norm=True,
                                               device=self.device, only_action=self.only_action)
        else:
            raise NotImplementedError
        
    def BCEloss(self,output, label, loss_manager, weight, name): 
        criterion = nn.BCELoss(reduction='sum')   
        err = criterion(output, label)
        loss_manager.addToLosses(name, weight, err)
        return weight*err
        
    def decode(self, z, l, t, only_action):
        l = l.long().to('cpu')
        t = t.to('cpu')
        return self.generator.forward_cgan(z,l,t, only_action)

    def train_on_batch(self, obs, label ,target_pos, epoch_batches,figdir,epoch,fixed_sample_state,
                       fixed_sample_label,fixed_sample_target_pos,
                       optimizer_D, optimizer_G,
                       loss_manager_D, loss_manager_G,
                       valid_mode=False,
                       device=torch.device('cpu'), label_smoothing=False,add_noise=False,only_action=False,minibatch=10):
                       
        batch_size = obs.size(0)       
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        optimizer_D.zero_grad()
        loss_manager_D.resetLosses()
        
        # train with real
        if label_smoothing:
            #real_label = torch.FloatTensor(batch_size,).uniform_(0.7,1.2).to(device)   
            real_label = torch.full((batch_size,), 0.9, device=device)
        else:
            real_label = torch.full((batch_size,), 1, device=device)
        if add_noise:
            obs = obs+torch.from_numpy(np.random.normal(0,0.001,obs.size())).type(torch.FloatTensor).to(device)
            obs = obs.to(device)
        else:
            obs = obs.to(device)
        label = label.to(device)
        target_pos = target_pos.to(device)
        output = self.discriminator.forward_cgan(obs, label, target_pos, only_action)
        D_x = output.mean().item()
        b=self.BCEloss(output,real_label,loss_manager_D,1.0,'loss_D_real')
             
        # train with fake
        sample_state = torch.randn(batch_size, self.state_dim, device=device)
        sample_label = (torch.rand(batch_size, ) * self.label_dim).type(torch.LongTensor)
        sample_t_pos = sample_target_pos(batch_size,TARGET_MAX_X, TARGET_MIN_X, TARGET_MAX_Y, TARGET_MIN_Y)
        fake_obs = self.generator.forward_cgan(sample_state, sample_label, sample_t_pos, only_action)
        if add_noise:
            fake_obs = fake_obs+torch.from_numpy(np.random.normal(0,0.001,fake_obs.size())).type(torch.FloatTensor).to(device)
            fake_obs = fake_obs.to(device)
        fake_label = torch.full((batch_size,), 0, device=device)   
        output = self.discriminator.forward_cgan(fake_obs.detach(), sample_label, sample_t_pos, only_action)
        
        D_G_z = output.mean().item()
        a=self.BCEloss(output, fake_label, loss_manager_D, 1.0, 'loss_D_fake')

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizer_G.zero_grad()
        loss_manager_G.resetLosses()
        output = self.discriminator.forward_cgan(fake_obs, sample_label, sample_t_pos, only_action)
        self.BCEloss(output,real_label,loss_manager_G,1.0,'loss_G')
        
        if not valid_mode:
            loss_manager_D.updateLossHistory()
            loss_manager_G.updateLossHistory()
        errD = loss_manager_D.computeTotalLoss()
        errG = loss_manager_G.computeTotalLoss()
        if not valid_mode:
            errD.backward()
            optimizer_D.step()
            
            errG.backward()
            optimizer_G.step()
            history_message = "D(x): {:.4f} D(G(z)): {:.4f}".format(D_x, D_G_z)
        else:
            history_message = " "
        errD = errD.item()
        errG = errG.item()
        
        if epoch_batches % (minibatch-1) == 0 and not valid_mode and figdir is not None :
            fake_obs = self.generator.forward_cgan(fixed_sample_state.to(device), fixed_sample_label, fixed_sample_target_pos, only_action)
            vutils.save_image(fake_obs[0].detach(),
                    '{}/fake_samples_epoch_{:03d}.png'.format(figdir, epoch+1),
                    normalize=True)
        return errD, errG, history_message
        
        


