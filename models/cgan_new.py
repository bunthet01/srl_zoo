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

# The DiscriminatorDC and GeneratorDC is inspired by https://github.com/pytorch/examples/tree/master/dcgan
class DiscriminatorDC(nn.Module):
    def __init__(self, state_dim, label_dim, img_shape):
        super(DiscriminatorDC, self).__init__()
        assert img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        #ndf = img_shape[1]*2
        ndf = img_shape[1]
        nc = int(img_shape[0])
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
    def __init__(self, state_dim, label_dim, img_shape):
        super(GeneratorDC, self).__init__()
        assert img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        nz = state_dim
        # ngf = img_shape[1]*2
        ngf = img_shape[1]
        nc = img_shape[0]
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
    def __init__(self, state_dim, label_dim, img_shape):
        super().__init__()
        self.state_dim = state_dim
        self.img_shape = img_shape
        self.label_dim = label_dim
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
        

    
    def build_model(self, model_type='custom_cnn'):
        assert model_type in ['custom_cnn', 'linear', 'mlp']
        self.generator = GeneratorDC(self.state_dim, self.label_dim, self.img_shape)
        self.discriminator = DiscriminatorDC(self.state_dim, self.label_dim, self.img_shape)
        
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)
        
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
                       device=torch.device('cpu')):
                       
        batch_size = obs.size(0)       
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        optimizer_D.zero_grad()
        loss_manager_D.resetLosses()
        
        # train with real
        real_label = torch.full((batch_size,), 1, device=device)
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
        
        if epoch_batches % 20 == 0 and not valid_mode and figdir is not None :
            vutils.save_image(obs[0],
                    '{}/real_obs_epoch_{:03d}.png'.format(figdir, epoch+1),
                    normalize=True)
            fake_obs = self.generator.forward_cgan(fixed_sample_state.to(device), self.one_hot[fixed_sample_label].to(device))
            vutils.save_image(fake_obs[0].detach(),
                    '{}/fake_samples_epoch_{:03d}.png'.format(figdir, epoch+1),
                    normalize=True)
        return errD, errG, history_message
        
        


