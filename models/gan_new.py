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
    from models.base_models import BaseModelSRL,BaseModelAutoEncoder, ConvSN2d, ConvTransposeSN2d, LinearSN, UNet
    from models.base_trainer import BaseTrainer
    from losses.losses import ganNonSaturateLoss, autoEncoderLoss, ganBCEaccuracy, AEboundLoss
from torchsummary import summary

class Discriminator(nn.Module):
    def __init__(self, state_dim, img_shape,
                 spectral_norm=False,
                 d_chs=16):  # 32
        super().__init__()
        assert img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        self.img_shape = img_shape
        self.state_dim = state_dim
        self.spectral_norm = spectral_norm
        self.d_chs = d_chs
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
            d_layer(start_chs, self.d_chs, spectral_norm=self.spectral_norm))       #torch.Size([4, 16, 32, 32], (bs=4, img_shape=(3,64,64), state_dim=200)
        self.modules_list.extend(
            d_layer(self.d_chs, self.d_chs*2, spectral_norm=self.spectral_norm))    #torch.Size([4, 32, 16, 16])
        self.modules_list.extend(
            d_layer(self.d_chs*2, self.d_chs*4, spectral_norm=self.spectral_norm))  #torch.Size([4, 64, 8, 8])
        self.modules_list.extend(
            d_layer(self.d_chs*4, self.d_chs*8, spectral_norm=self.spectral_norm))  #torch.Size([4, 128, 4, 4]) 
        self.modules_list.extend(
            d_layer(self.d_chs*8, self.d_chs*8, spectral_norm=self.spectral_norm))  #torch.Size([4, 128, 2, 2])

        if self.spectral_norm:
            self.modules_list.append(ConvSN2d(self.d_chs*8, self.d_chs*4,
                                              kernel_size=3, stride=1, padding=1))  #torch.Size([4, 64, 2, 2])

            last_channels = self.modules_list[-1].out_channels                      #64
            times = COUNT_IMG_REDUCE
            in_features = last_channels * \
                (self.img_shape[1]//2**times) * (self.img_shape[2]//2**times)       #256
            self.before_last = LinearSN(in_features, self.state_dim, bias=True)
            self.last = LinearSN(self.state_dim, 1, bias=True)
        else:
            self.modules_list.append(nn.Conv2d(self.d_chs*8, self.d_chs*4,
                                               kernel_size=3, stride=1, padding=1)) 
            last_channels = self.modules_list[-1].out_channels          
            times = COUNT_IMG_REDUCE
            in_features = last_channels * \
                (self.img_shape[1]//2**times) * (self.img_shape[2]//2**times)
            self.before_last = nn.Linear(
                in_features, self.state_dim, bias=True)
            self.last = nn.Linear(self.state_dim, 1, bias=True)

    def forward(self, x):
        for layer in self.modules_list:
            x = layer(x)
        x = x.view(x.size(0), -1)  # flatten    #torch.Size([4, 256])
        x = self.activations['lrelu'](x)        #torch.Size([4, 256])
        x = self.before_last(x)                 #torch.Size([4, 200])
        x = self.activations['lrelu'](x)        #torch.Size([4, 200])
        x = self.last(x)                        #torch.Size([4, 1])
        x = self.activations['sigmoid'](x)      #torch.Size([4, 1])
        return x
class GeneratorUnet(nn.Module):
    def __init__(self, state_dim, img_shape,
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
                         batch_norm=self.unet_bn, spec_norm=self.spectral_norm, dropout=self.unet_drop, up_mode='upconv', out_ch=out_channels)
        prev_channels = self.unet.out_ch
        if self.spectral_norm:
            self.last = ConvSN2d(prev_channels, out_channels,
                                 kernel_size=3, stride=1, padding=1)
        else:
            self.last = nn.Conv2d(
                prev_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.squeeze()
        x = self.first(x)
        x = self.activations['lrelu'](x)
        x = x.view(x.size(0), *self.img_shape)
        x = self.unet(x)
        x = self.last(x)
        x = self.activations['tanh'](x)
        return x

class GeneratorResNet(BaseModelAutoEncoder):
    """
    ResNet Generator
    """

    def __init__(self, state_dim, img_shape, spectral_norm=False):
        super(GeneratorResNet, self).__init__(state_dim=state_dim, img_shape=img_shape)
        assert img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        self.state_dim = state_dim
        self.img_shape = img_shape
        self.spectral_norm = spectral_norm
        
        outshape = summary(self.encoder_conv, self.img_shape, show=False) # [-1, channels, high, width] # [-1,64,1,1]
        self.img_height, self.img_width = outshape[-2:]
        self.decoder_fc = nn.Sequential(
            nn.Linear(state_dim, self.img_height*self.img_width*64))
        print("Generator: sn = ", spectral_norm)
        if spectral_norm:
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

    def forward(self, z):
        z = z.squeeze()
        decoded = self.decoder_fc(z)
        decoded = decoded.view(z.size(0),64, self.img_height,self.img_width)
        return self.decoder_conv(decoded)



# The DiscriminatorDC and GeneratorDC is inspired by https://github.com/pytorch/examples/tree/master/dcgan
class DiscriminatorDC(nn.Module):
    def __init__(self, state_dim, img_shape, spectral_norm =False):
        super(DiscriminatorDC, self).__init__()
        self.img_shape = img_shape
        self.state_dim = state_dim
        print("Discriminator: sn = ", spectral_norm)
        assert self.img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        ndf = 128
        nc = img_shape[0]
        if spectral_norm:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                ConvSN2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
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
            )

        else:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
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
            )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)
        
class GeneratorDC(nn.Module):
    def __init__(self, state_dim, img_shape, spectral_norm=False):
        super(GeneratorDC, self).__init__()
        self.img_shape = img_shape
        self.state_dim = state_dim
        print("Generator: sn = ", spectral_norm)
        assert self.img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        nz = state_dim
        ngf = 128
        nc = img_shape[0]
        if spectral_norm:
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                ConvTransposeSN2d(     nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
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
                ConvTransposeSN2d(    ngf,      nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        else:
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
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
                nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

    def forward(self, input):
        output = self.main(input)
        return output      
        

class GanNewTrainer(BaseTrainer):
    def __init__(self, state_dim, img_shape):
        super().__init__()
        self.state_dim = state_dim
        self.img_shape = img_shape
    
    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
            
    def build_model(self, model_type='dc'):
        assert model_type in ['custom_cnn', 'dc', 'unet']
        if model_type == "dc":
            self.generator = GeneratorDC(self.state_dim, self.img_shape, spectral_norm=False)
            self.discriminator = DiscriminatorDC(self.state_dim, self.img_shape, spectral_norm=True)
            
            self.generator.apply(self.weights_init)
            self.discriminator.apply(self.weights_init)
        elif model_type == "custom_cnn":
            self.generator = GeneratorResNet(self.state_dim, self.img_shape, spectral_norm=True)
            self.discriminator = Discriminator(self.state_dim, self.img_shape, spectral_norm=True)
        elif model_type == "unet":
            self.generator = GeneratorUnet(self.state_dim, self.img_shape, spectral_norm=True)
            self.discriminator = Discriminator(self.state_dim, self.img_shape, spectral_norm=True)
        else:
            raise NotImplementedError
        
        
    def BCEloss(self,output, label, loss_manager, weight, name): 
        criterion = nn.BCELoss(reduction='sum')   
        err = criterion(output, label)
        loss_manager.addToLosses(name, weight, err)
        return weight*err
        
    def decode(self, z):
        z = z.view(z.size(0),z.size(1),1,1)
        return self.generator(z)
    
        
    def train_on_batch(self, obs ,epoch_batches,figdir,epoch,fixed_sample_state,
                       optimizer_D, optimizer_G,
                       loss_manager_D, loss_manager_G,
                       valid_mode=False,
                       device=torch.device('cpu'), label_smothing=False, add_noise=False,minibatch=10):
        batch_size = obs.size(0)       
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        optimizer_D.zero_grad()
        loss_manager_D.resetLosses()
        
        # train with real
        if label_smothing:
            # real_label = torch.FloatTensor(batch_size,).uniform_(0.7,1.2).to(device)
            real_label = torch.full((batch_size,), 0.9, device=device)
        else:
            real_label = torch.full((batch_size,), 1, device=device)
        if add_noise:
            obs = obs+torch.from_numpy(np.random.normal(0,0.001,obs.size())).type(torch.FloatTensor).to(device)
        else:
            obs = obs.to(device)
        output = self.discriminator(obs)
        D_x = output.mean().item()
        self.BCEloss(output,real_label,loss_manager_D,1.0,'loss_D_real')
             
        # train with fake
        sample_state = torch.randn(batch_size, self.state_dim, 1, 1, device=device)
        fake_label = torch.full((batch_size,), 0, device=device)
        fake_obs = self.generator(sample_state)
        if add_noise:
            fake_obs = fake_obs+torch.from_numpy(np.random.normal(0,0.001,fake_obs.size())).type(torch.FloatTensor).to(device)

        output = self.discriminator(fake_obs.detach())
        D_G_z = output.mean().item()
        self.BCEloss(output, fake_label, loss_manager_D, 1.0, 'loss_D_fake')

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizer_G.zero_grad()
        loss_manager_G.resetLosses()
        output = self.discriminator(fake_obs)
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
        
        if epoch_batches % (minibatch-1) == 0 and not valid_mode:
            #vutils.save_image(obs[0],
            #        '{}/real_obs_epoch_{:03d}.png'.format(figdir, epoch+1),
            #        normalize=True)
            fake_obs = self.generator(fixed_sample_state.to(device))
            vutils.save_image(fake_obs[0].detach(),
                    '{}/fake_samples_epoch_{:03d}.png'.format(figdir, epoch+1),
                    normalize=True)
        return errD, errG, history_message
        
        


