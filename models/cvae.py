from __future__ import print_function, division, absolute_import
from .models import *


class CNNCVAE(BaseModelVAE):
    def __init__(self, state_dim=3, class_dim=1):
        super(CNNCVAE, self).__init__()
        self.state_dim = state_dim
        self.class_dim = class_dim

        self.encoder_f1 = nn.Linear(6 * 6 * 64+self.class_dim, self.state_dim)
        self.encoder_f2 = nn.Linear(6 * 6 * 64+self.class_dim, self.state_dim)

        self.decode_fc = nn.Linear(self.state_dim+self.class_dim, 6*6*64)

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

    def forward_cvae(self, x, c):
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

    def getStates_cvae(self, x, c):
        """
        :param x: (th.Tensor) observation
        :param c: (th.Tensor)
        :return: (th.Tensor)
        """
        mu, logvar = self.encode_cvae(x, c)
        return self.reparameterize(mu, logvar)
    
    







