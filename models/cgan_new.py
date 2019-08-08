import numpy as np
import torch
from torch import nn
import torchvision.utils as vutils
import torch.nn.functional as F
try:
    # relative import
    from .base_models import BaseModelSRL, ConvSN2d, ConvTransposeSN2d, LinearSN, UNet
    from .base_trainer import BaseTrainer
    from ..losses.losses import ganNonSaturateLoss, autoEncoderLoss, ganBCEaccuracy, AEboundLoss
except:
    from models.base_models import BaseModelSRL, ConvSN2d, ConvTransposeSN2d, LinearSN, UNet
    from models.base_trainer import BaseTrainer
    from losses.losses import ganNonSaturateLoss, autoEncoderLoss, ganBCEaccuracy, AEboundLoss
from torchsummary import summary

class GeneratorUnet(nn.Module):
    def __init__(self, state_dim, img_shape, label_dim,
                 unet_depth=2,  # 3
                 unet_ch=16,  # 32
                 spectral_norm=False, device='cpu',
                 unet_bn=False,
                 unet_drop=0.0):
        super().__init__()
        self.state_dim = state_dim
        self.img_shape = img_shape
        self.device = device
        self.label_dim = label_dim
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
                self.state_dim + self.label_dim, np.prod(self.img_shape), bias=True)
        else:
            self.first = nn.Linear(
                self.state_dim + self.label_dim, np.prod(self.img_shape), bias=True)

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

    def forward_cgan(self, x, l):
        x = torch.cat([x, one_hot(l).to(self.device)], 1)
        x = self.first(x)
        x = self.activations['lrelu'](x)
        x = x.view(x.size(0), *self.img_shape)
        x = self.unet(x)
        x = self.last(x)
        x = self.activations['tanh'](x)
        return x


class GeneratorResNet(BaseModelSRL):
    """
    ResNet Generator
    """

    def __init__(self, state_dim, img_shape, label_dim, in_shape, spectral_norm=False, device='cpu'):
        super().__init__(state_dim=state_dim, img_shape=img_shape)
        assert img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        self.state_dim = state_dim
        self.in_shape = in_shape
        self.img_shape = img_shape
        self.device = device
        self.label_dim = label_dim
        self.spectral_norm = spectral_norm

        _, self.in_height, self.in_width = self.in_shape  ## [channel, high, width]

        self.decoder_fc = nn.Sequential(
            nn.Linear(state_dim + label_dim, self.in_height * self.in_width * 64)
        )
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

    def forward_cgan(self, x, l):
        decoded = torch.cat([x, one_hot(l).to(self.device)], 1)
        decoded = self.decoder_fc(decoded)
        decoded = decoded.view(x.size(0), 64, self.in_height, self.in_width)
        return self.decoder_conv(decoded)
class Discriminator(nn.Module):
    def __init__(self, state_dim, img_shape,label_dim,
                 spectral_norm=False,device='cpu',
                 d_chs=16):  # 32
        super().__init__()
        self.img_shape = img_shape
        self.state_dim = state_dim
        self.label_dim = label_dim
        self.device = device
        self.spectral_norm = spectral_norm
        self.d_chs = d_chs

        assert self.img_shape[0] < 10, "Pytorch uses 'channel first' convention."
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

            last_channels = self.modules_list[-1].out_channels   #64
            times = COUNT_IMG_REDUCE
            in_features = last_channels * \
                (self.img_shape[1]//2**times) * (self.img_shape[2]//2**times)   
            self.before_last = LinearSN(in_features, self.state_dim, bias=True)
            self.last = LinearSN(self.state_dim+self.label_dim, 1, bias=True)
        else:
            self.modules_list.append(nn.Conv2d(self.d_chs*8, self.d_chs*4,
                                               kernel_size=3, stride=1, padding=1))
            last_channels = self.modules_list[-1].out_channels
            times = COUNT_IMG_REDUCE
            in_features = last_channels * \
                (self.img_shape[1]//2**times) * (self.img_shape[2]//2**times)
            self.before_last = nn.Linear(
                in_features, self.state_dim, bias=True)
            self.last = nn.Linear(self.state_dim+self.label_dim, 1, bias=True)

    def forward_cgan(self, x, l):
        for layer in self.modules_list:
            x = layer(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.activations['lrelu'](x)
        x = self.before_last(x)
        x = self.activations['lrelu'](x)
        x = torch.cat([x,one_hot(l).to(self.device)],1)
        x = self.last(x)
        x = self.activations['sigmoid'](x)
        return x

# The DiscriminatorDC and GeneratorDC is inspired by https://github.com/pytorch/examples/tree/master/dcgan
class DiscriminatorDC(nn.Module):
    def __init__(self, state_dim, label_dim, img_shape,spectral_norm=False):
        super(DiscriminatorDC, self).__init__()
        assert img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        print("Discriminator: sn = ", spectral_norm)
        ndf = 128
        nc = int(img_shape[0])
        if spectral_norm:
            self.conv1_1 = nn.Sequential(
                # input is (nc) x 64 x 64
                ConvSN2d(nc, int(ndf/2), 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True))
            self.conv1_2 = nn.Sequential(
                # input is (label_dim) x 1 x 1
                ConvSN2d(label_dim, int(ndf/2), 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                )
            self.conv2 = nn.Sequential(
                # state size. (ndf) x 32 x 32
                ConvSN2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                ConvSN2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                ConvSN2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                ConvSN2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
                # state size. (1x1x1)

                )
        else:
            self.conv1_1 = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, int(ndf/2), 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True))
            self.conv1_2 = nn.Sequential(
                # input is (label_dim) x 1 x 1
                nn.Conv2d(label_dim, int(ndf/2), 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                )
            self.conv2 = nn.Sequential(
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
                # state size. (1x1x1)

                )

    def forward_cgan(self, input, label):
        x = self.conv1_1(input)
        y = self.conv1_2(label)
        x = torch.cat([x,y],1)
        x = self.conv2(x)
        
        return x.view(-1, 1).squeeze(1)
        
class GeneratorDC(nn.Module):
    def __init__(self, state_dim, label_dim, img_shape, spectral_norm=False):
        super(GeneratorDC, self).__init__()
        assert img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        print("Generator: sn = ", spectral_norm)
        nz = state_dim
        ngf = 128
        nc = img_shape[0]
        if spectral_norm:
            self.deconv1_1 = nn.Sequential(
                # input is Z, going into a convolution
                ConvTransposeSN2d(nz, ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True))
            self.deconv1_2 = nn.Sequential(
                # input is Z, going into a convolution
                ConvTransposeSN2d(label_dim,ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True))
            self.deconv2 = nn.Sequential(
                # state size. (ngf*8) x 4 x 4
                ConvTransposeSN2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                ConvTransposeSN2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                ConvTransposeSN2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                ConvTransposeSN2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64)
                ) 
        else:
            self.deconv1_1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True))
            self.deconv1_2 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(label_dim,ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True))
            self.deconv2 = nn.Sequential(
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64)
                )   
    def forward_cgan(self,input, label):
        x = self.deconv1_1(input)
        y = self.deconv1_2(label)
        x = torch.cat([x,y],1)
        x = self.deconv2(x)
        return x      
        

class CGanNewTrainer(BaseTrainer):
    def __init__(self, state_dim, label_dim, img_shape,device):
        super().__init__()
        self.state_dim = state_dim
        self.img_shape = img_shape
        self.label_dim = label_dim
        self.device = device
        self.one_hot = torch.zeros(self.label_dim, self.label_dim)
        self.one_hot = self.one_hot.scatter(1, torch.arange(self.label_dim).type(torch.LongTensor).view(self.label_dim,1), 1).view(self.label_dim, self.label_dim, 1, 1)
        self.fill = torch.zeros([self.label_dim, self.label_dim, self.img_shape[1], self.img_shape[2]])
        for i in range(self.label_dim):
            self.fill[i, i, :, :] = 1
    
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
            self.generator = GeneratorDC(self.state_dim, self.label_dim, self.img_shape, spectral_norm=False)
            self.discriminator = DiscriminatorDC(self.state_dim, self.label_dim, self.img_shape, spectral_norm=True)

            self.generator.apply(self.weights_init)
            self.discriminator.apply(self.weights_init)
        elif model_type == "unet":
            self.generator = GeneratorUnet(self.state_dim, self.img_shape, self.label_dim, spectral_norm=True,
                                           device=self.device)
            self.discriminator = Discriminator(self.state_dim, self.img_shape, self.label_dim, spectral_norm=True,
                                               device=self.device)
        elif model_type == "custom_cnn":
            self.generator = GeneratorResNet(self.state_dim, self.img_shape, self.label_dim,
                                             spectral_norm=True, device=self.device)
            self.discriminator = Discriminator(self.state_dim, self.img_shape, self.label_dim, spectral_norm=True,
                                               device=self.device)
        else:
            raise NotImplementedError
        
    def BCEloss(self,output, label, loss_manager, weight, name): 
        criterion = nn.BCELoss(reduction='sum')   
        err = criterion(output, label)
        loss_manager.addToLosses(name, weight, err)
        return weight*err
    def decode(self, z, l):
        z = z.view(z.size(0),z.size(1),1,1)
        l = l.view(l.size(0),l.size(1),1,1)
        return self.generator.forward_cgan(z,l)

    def train_on_batch(self, obs, label ,epoch_batches,figdir,epoch,fixed_sample_state,
                       fixed_sample_label,
                       optimizer_D, optimizer_G,
                       loss_manager_D, loss_manager_G,
                       valid_mode=False,
                       device=torch.device('cpu'), label_smothing=False,add_noise=False,minibatch=10):
                       
        batch_size = obs.size(0)       
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        optimizer_D.zero_grad()
        loss_manager_D.resetLosses()
        
        # train with real
        if label_smothing:
            real_label = torch.FloatTensor(batch_size,).uniform_(0.7,1.2).to(device)
        else:
            real_label = torch.full((batch_size,), 1, device=device)
        if add_noise:
            obs = obs+torch.from_numpy(np.random.normal(0,0.001,obs.size())).type(torch.FloatTensor).to(device)
            obs = obs.to(device)
        else:
            obs = obs.to(device)
        label = self.fill[label].to(device)
        output = self.discriminator.forward_cgan(obs, label)
        D_x = output.mean().item()
        b=self.BCEloss(output,real_label,loss_manager_D,1.0,'loss_D_real')
             
        # train with fake
        sample_state = torch.randn(batch_size, self.state_dim, 1, 1, device=device)
        sample_label = (torch.rand(batch_size, 1) * self.label_dim).type(torch.LongTensor).squeeze()	
        one_hot_sample_label = self.one_hot[sample_label].to(device)
        fake_obs = self.generator.forward_cgan(sample_state, one_hot_sample_label)
        if add_noise:
            fake_obs = fake_obs+torch.from_numpy(np.random.normal(0,0.001,fake_obs.size())).type(torch.FloatTensor).to(device)
            fake_obs = fake_obs.to(device)
        fake_label = torch.full((batch_size,), 0, device=device)
        
        output = self.discriminator.forward_cgan(fake_obs.detach(), self.fill[sample_label].to(device))
        D_G_z = output.mean().item()
        a=self.BCEloss(output, fake_label, loss_manager_D, 1.0, 'loss_D_fake')

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizer_G.zero_grad()
        loss_manager_G.resetLosses()
        output = self.discriminator.forward_cgan(fake_obs, self.fill[sample_label].to(device))
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
            # vutils.save_image(obs[0],
            #         '{}/real_obs_epoch_{:03d}.png'.format(figdir, epoch+1),
            #         normalize=True)
            fake_obs = self.generator.forward_cgan(fixed_sample_state.to(device), self.one_hot[fixed_sample_label].to(device))
            vutils.save_image(fake_obs[0].detach(),
                    '{}/fake_samples_epoch_{:03d}.png'.format(figdir, epoch+1),
                    normalize=True)
        return errD, errG, history_message
        
        


