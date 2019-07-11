import numpy as np
import torch
from torch import nn
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
        x = self.first(x)
        x = self.activations['lrelu'](x)
        x = x.view(x.size(0), *self.img_shape)
        x = self.unet(x)
        x = self.last(x)
        x = self.activations['tanh'](x)
        return x


class Discriminator(nn.Module):
    def __init__(self, state_dim, img_shape,
                 spectral_norm=False,
                 d_chs=16):  # 32
        super().__init__()
        self.img_shape = img_shape
        self.state_dim = state_dim
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

            last_channels = self.modules_list[-1].out_channels
            times = COUNT_IMG_REDUCE
            in_features = last_channels * \
                (self.img_shape[1]//2**times) * (self.img_shape[2]//2**times)
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
        x = x.view(x.size(0), -1)  # flatten
        x = self.activations['lrelu'](x)
        x = self.before_last(x)
        x = self.activations['lrelu'](x)
        x = self.last(x)
        x = self.activations['sigmoid'](x)
        return x


class EncoderUnet(BaseModelSRL):
    """

    Note: Only EncoderUnet has getStates method.
    """

    def __init__(self, state_dim, img_shape,
                 unet_depth=2,  # 3
                 unet_ch=16,
                 unet_bn=False,
                 unet_drop=0.0,
                 spectral_norm=False):
        super().__init__(state_dim=state_dim, img_shape=img_shape)
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
            # self.top_model = nn.Sequential(OrderDict([
            #                     ('conv1', nn.Conv2d(prev_channels, 1, kernel_size=4, stride=2, padding=1)),
            #                     ('relu1', self.activations['relu']),
            #                     ('conv2', nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1)),
            #                     ('relu2', self.activations['relu']),
            #                     ('dense1', nn.Linear(inter_features, 100, bias=True)),
            #                     ('relu3', self.activations['relu']),
            #                     ('dense2', nn.Linear(100, self.state_dim, bias=True)),
            #                 ]))

    def forward(self, x):
        x = self.unet(x)
        for layer in self.modules_list:
            x = layer(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.before_last(x)
        x = self.activations['lrelu'](x)
        x = self.last(x)
        return x


class EncoderResNet(BaseModelSRL):
    """

    Note: Only Encoder has getStates method.
    """

    def __init__(self, state_dim, img_shape,
                 spectral_norm=False):
        super().__init__(state_dim=state_dim, img_shape=img_shape)
        assert img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        self.state_dim = state_dim
        self.img_shape = img_shape
        self.spectral_norm = spectral_norm

        if self.spectral_norm:
            self.encoder_conv = nn.Sequential(
                ConvSN2d(self.img_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=True),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                ConvSN2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),  # TODO TODO bias TRUE?
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),

                ConvSN2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),  # TODO TODO bias TRUE?
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2)
            )

        else:
            self.encoder_conv = nn.Sequential(
                nn.Conv2d(self.img_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=True),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),  # TODO TODO bias TRUE?
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),

                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),  # TODO TODO bias TRUE?
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2)
            )
        outshape = summary(self.encoder_conv, self.img_shape, show=False)  # [-1, channels, high, width]
        self.img_height, self.img_width = outshape[-2:]
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.img_height * self.img_width * 64, self.state_dim)
        )

    def forward(self, x):
        encoded = self.encoder_conv(x)
        encoded = encoded.view(encoded.size(0), -1)
        return self.encoder_fc(encoded)


class GeneratorResNet(BaseModelSRL):
    """
    ResNet Generator
    """

    def __init__(self, state_dim, img_shape, in_shape, spectral_norm=False):
        super().__init__(state_dim=state_dim, img_shape=img_shape)
        assert img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        self.state_dim = state_dim
        self.in_shape = in_shape
        self.img_shape = img_shape
        self.spectral_norm = spectral_norm

        _, self.in_height, self.in_width = self.in_shape ## [channel, high, width]

        self.decoder_fc = nn.Sequential(
            nn.Linear(state_dim, self.in_height * self.in_width * 64)
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
        

    def forward(self, x):
        decoded = self.decoder_fc(x)
        decoded = decoded.view(x.size(0), 64, self.in_height, self.in_width)
        return self.decoder_conv(decoded)


class GANTrainer(BaseTrainer):
    def __init__(self, state_dim=2, img_shape=None):
        super().__init__()
        self.img_shape = img_shape
        self.state_dim = state_dim

    def build_model(self, model_type='unet'):
        if model_type == 'unet':
            self.encoder = EncoderUnet(self.state_dim, self.img_shape, spectral_norm=False)
            self.generator = GeneratorUnet(self.state_dim, self.img_shape, spectral_norm=True)
            self.discriminator = Discriminator(self.state_dim, self.img_shape, spectral_norm=True)
        elif model_type == 'custom_cnn':
            self.encoder = EncoderResNet(self.state_dim, self.img_shape, spectral_norm=False)
            outshape = summary(self.encoder.encoder_conv, self.img_shape, show=False) # [-1, channels, high, width]
            self.generator = GeneratorResNet(self.state_dim, self.img_shape, outshape[1:], spectral_norm=True)
            self.discriminator = Discriminator(self.state_dim, self.img_shape, spectral_norm=True)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.encoder(x)

    def train_on_batch_E(self, obs, next_obs, optimizer, loss_manager, valid_mode=False, device=torch.device('cpu')):
        """
        loss_manager will cumulate the loss (pytorch tensor) of e.g. inverse/forward/reward models, etc.

        """
        # Compute loss tensor (add to the previous losses cumulated in loss_manager)
        # state_pred, state_pred_next = self.encoder(obs), self.encoder(next_obs)
        # reconstruct_obs, reconstruct_obs_next = self.generator(state_pred), self.generator(state_pred_next)

        # reconstruct_obs, reconstruct_obs_next = self.reconstruct(obs), self.reconstruct(next_obs)

        state_pred = self.encoder(obs)
        reconstruct_obs = self.generator(state_pred)
        reconstruct_obs_next = self.reconstruct(next_obs)
        autoEncoderLoss(obs, reconstruct_obs, next_obs, reconstruct_obs_next, 10000.0, loss_manager)
        AEboundLoss(state_pred, 1.0, loss_manager)
        loss = self.update_nn_weights(optimizer, loss_manager, valid_mode=valid_mode)
        return loss

    def train_on_batch_D(self, obs, label_valid, label_fake, optimizer, loss_manager, valid_mode=False, device=torch.device('cpu')):
        sample_state = torch.randn((obs.size(0), self.state_dim), requires_grad=False).to(device)
        fake_img = self.generator(sample_state)
        # fake_loss
        fake_img_rating = self.discriminator(fake_img.detach())
        ganNonSaturateLoss(fake_img_rating, label_fake, weight=1.0, loss_manager=loss_manager, name="ns_loss_D_fake")
        # real_loss
        real_img_rating = self.discriminator(obs)
        ganNonSaturateLoss(real_img_rating, label_valid, weight=1.0, loss_manager=loss_manager, name="ns_loss_D_real")
        loss = self.update_nn_weights(optimizer, loss_manager, valid_mode=valid_mode)
        acc_pos = ganBCEaccuracy(real_img_rating, label=1)
        acc_neg = ganBCEaccuracy(fake_img_rating, label=0)
        acc = (acc_pos + acc_neg) / 2.
        return loss, acc.item()

    def train_on_batch_G(self, obs, label_valid, optimizer, loss_manager, valid_mode=False, device=torch.device('cpu')):
        sample_state = torch.randn((obs.size(0), self.state_dim), requires_grad=False).to(device)
        fake_img = self.generator(sample_state)
        fake_rating = self.discriminator(fake_img)
        ganNonSaturateLoss(fake_rating, label_valid, weight=1.0, loss_manager=loss_manager, name="ns_loss_G")
        loss = self.update_nn_weights(optimizer, loss_manager, valid_mode=valid_mode)
        acc = ganBCEaccuracy(fake_rating, label=1)
        return loss, acc.item()

    def reconstruct(self, x):
        return self.generator(self.encoder(x))

    def train_on_batch(self, obs, next_obs,
                       optimizer_D, optimizer_G, optimizer_E,
                       loss_manager_D, loss_manager_G, loss_manager_E,
                       epoch_batches_D, epoch_batches_G, epoch_batches_E,
                       epoch_loss_D, epoch_loss_G, epoch_loss_E,
                       d_acc, g_acc,
                       batch_size=32,
                       dataloader=None,
                       valid_mode=False,
                       device=torch.device('cpu')):
        # import ipdb; ipdb.set_trace()
        label_valid = torch.ones((batch_size, 1)).to(device)
        label_fake = torch.zeros((batch_size, 1)).to(device)
        if not valid_mode:
            # === Train the Discriminator ===
            D_steps = 3 if (d_acc < 0.8) else 1
            G_steps = 3 if (g_acc < 0.2) else 1
            acc_cum = 0
            for _ in range(D_steps):
                optimizer_D.zero_grad()
                loss_manager_D.resetLosses()
                (sample_idx, obs, next_obs, action, reward, cls_gt) = next(dataloader)
                cls_gt = cls_gt.to(device)
                obs = obs.to(device)
                # re-define the length label_valid/label_fake, because obs.size(0) changes
                d_loss, d_acc = self.train_on_batch_D(obs, label_valid[:obs.size(0)], label_fake[:obs.size(
                    0)], optimizer_D, loss_manager_D, valid_mode=valid_mode, device=device)
                epoch_loss_D += d_loss
                epoch_batches_D += 1
                acc_cum += d_acc
            d_acc = acc_cum / D_steps

            # === Train the Generator ===
            acc_cum = 0
            for _ in range(G_steps):
                optimizer_G.zero_grad()
                loss_manager_G.resetLosses()
                (sample_idx, obs, next_obs, action, reward, cls_gt) = next(dataloader)
                cls_gt = cls_gt.to(device)
                obs = obs.to(device)
                # re-define the length label_valid, because obs.size(0) changes
                g_loss, g_acc = self.train_on_batch_G(obs, label_valid[:obs.size(
                    0)], optimizer_G, loss_manager_G, valid_mode=valid_mode, device=device)
                epoch_loss_G += g_loss
                epoch_batches_G += 1
                acc_cum += g_acc
            g_acc = acc_cum / G_steps
        # === Train the Encoder and the other components (e.g. forward/inverse/reward model) ===
        E_steps = 10 if not valid_mode else 1
        for _ in range(E_steps):
            optimizer_E.zero_grad()
            loss_manager_E.resetLosses()
            (sample_idx, obs, next_obs, action, reward, cls_gt) = next(dataloader)
            obs, next_obs = obs.to(device), next_obs.to(device)
            cls_gt = cls_gt.to(device)
            e_loss = self.train_on_batch_E(
                obs, next_obs, optimizer_E, loss_manager_E, valid_mode=valid_mode, device=device)
            epoch_loss_E += e_loss
            epoch_batches_E += 1
        if not valid_mode:
            train_loss_D = epoch_loss_D / float(epoch_batches_D)
            train_loss_G = epoch_loss_G / float(epoch_batches_G)
            train_loss = epoch_loss_E / float(epoch_batches_E)

        else:
            val_loss = epoch_loss_E / float(epoch_batches_E)

        
        if not valid_mode:
            loss = train_loss
            history_message = "D_loss: {:.4f} acc: {:.1%} G_loss: {:.4f} acc: {:.1%}".format(
                train_loss_D, d_acc, train_loss_G, g_acc)
        else:
            loss = val_loss
            # val_loss_str = "{:.2f}*".format(
            #     val_loss) if val_loss < best_error else "{:.2f}".format(val_loss)
            history_message = " "


        return loss, history_message


if __name__ == "__main__":
    print("Start")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torchsummary import summary  # requires torchsummary==1.5.2,
    # device = torch.device("cuda:0")
    # chs = 3
    # model = UNet(in_ch=chs, out_ch=3, start_ch=32, depth=3, batch_norm=False,
    #              spec_norm=False)  # .to(device)
    # a = 128
    # summary(model, (chs, a, a))

    # model = GeneratorUnet(10, img_shape=(3, 128, 128))
    # summary(model, (10,))

    # model = Discriminator(4, (3, 128, 128))
    # summary(model, (3, 128, 128))

    model = EncoderUnet(4, (3, 128, 128))
    summary(model, (3, 128, 128))
