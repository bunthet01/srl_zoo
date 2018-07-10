from .autoencoders import CNNAutoEncoder, DenseAutoEncoder, LinearAutoEncoder
from .vae import CNNVAE, DenseVAE
from .forward_inverse import BaseForwardModel, BaseInverseModel, BaseRewardModel
from .priors import SRLConvolutionalNetwork, SRLDenseNetwork, SRLLinear
from .triplet import EmbeddingNet
from .models import *

# In case of importing into the SRL repository
try:
    from preprocessing.preprocess import getInputDim
# In case of importing material from modules.py into the external Robotics RL repository,
# consider the relative path to the package
except ImportError:
    from ..preprocessing.preprocess import getInputDim


class SRLModules(BaseForwardModel, BaseInverseModel, BaseRewardModel):
    def __init__(self, state_dim=2, action_dim=6, cuda=False, model_type="custom_cnn", losses=None):
        """
        A model that can combine AE/VAE + Inverse + Forward + Reward models
        :param state_dim: (int)
        :param action_dim: (int)
        :param cuda: (bool)
        :param model_type: (str)
        :param losses: ([str])
        """
        self.model_type = model_type
        self.losses = losses
        BaseForwardModel.__init__(self)
        BaseInverseModel.__init__(self)
        BaseRewardModel.__init__(self)

        self.cuda = cuda

        self.initForwardNet(state_dim, action_dim)
        self.initInverseNet(state_dim, action_dim)
        self.initRewardNet(state_dim)

        # Architecture
        if model_type == "custom_cnn":
            if "autoencoder" in losses:
                self.model = CNNAutoEncoder(state_dim)
            elif "vae" in losses:
                self.model = CNNVAE(state_dim)
            else:
                # for losses not depending on specific architecture (supervised, inv, fwd..)
                self.model = CustomCNN(state_dim)

        elif model_type == "mlp":
            if "autoencoder" in losses:
                self.model = DenseAutoEncoder(input_dim=getInputDim(), state_dim=state_dim)
            elif "vae" in losses:
                self.model = DenseVAE(input_dim=getInputDim(),
                                      state_dim=state_dim)
            else:
                # for losses not depending on specific architecture (supervised, inv, fwd..)
                self.model = SRLDenseNetwork(getInputDim(), state_dim, cuda=cuda)

        elif model_type == "linear":
            if "autoencoder" in losses:
                self.model = LinearAutoEncoder(input_dim=getInputDim(), state_dim=state_dim)
            else:
                # for losses not depending on specific architecture (supervised, inv, fwd..)
                self.model = SRLLinear(input_dim=getInputDim(), state_dim=state_dim, cuda=cuda)

        elif model_type == "resnet":
            self.model = SRLConvolutionalNetwork(state_dim, cuda)

        if losses is not None and "triplet" in losses:
            # pretrained resnet18 with fixed weights
            self.model = EmbeddingNet(state_dim)

    def getStates(self, observations):
        """
        :param observations: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.model.getStates(observations)

    def forward(self, x):
        if self.model_type == 'linear' or self.model_type == 'mlp':
            x = x.contiguous()
        return self.model(x)

    def encode(self, x):
        if "triplet" in self.losses:
            return self.model(x)
        else:
            raise NotImplementedError()

    def forward_triplets(self, anchor, positive, negative):
        """
        Overriding the forward function in the case of Triplet loss
        anchor : observation
        positive : observation
        negative : observation
        """
        return self.model(anchor), self.model(positive), self.model(negative)


class SRLModulesSplit(BaseForwardModel, BaseInverseModel, BaseRewardModel):
    def __init__(self, state_dim=2, action_dim=6, cuda=False, model_type="custom_cnn", losses=None, split_index=1):
        """
        A model that can split representation, combining
        AE/VAE for the first split with Inverse + Forward in the second split
        Reward model is learned for all the dimensions
        :param state_dim: (int)
        :param action_dim: (int)
        :param cuda: (bool)
        :param model_type: (str)
        :param losses: ([str])
        :param split_index: (int) Number of dimensions for the first split
        """

        assert split_index < state_dim, "The second split must be of dim >= 1, consider increasing the state_dim or decreasing the split_index"
        assert "autoencoder" in losses or "vae" in losses, "You must use autoencoder/vae when splitting the representation"

        self.model_type = model_type
        self.losses = losses

        BaseForwardModel.__init__(self)
        BaseInverseModel.__init__(self)
        BaseRewardModel.__init__(self)

        self.cuda = cuda
        self.state_dim = state_dim

        # TODO: try with .detach() to give all the state to the decoder
        # but backpropagate only on part of it
        self.dim_first_method = split_index
        self.dim_second_method = state_dim - split_index
        self.first_split_indices = (slice(None, None), slice(None, split_index))  # [:, :split_index]
        self.second_split_indices = (slice(None, None), slice(split_index, None))  # [:, split_index:]

        self.initForwardNet(self.dim_second_method, action_dim)
        self.initInverseNet(self.dim_second_method, action_dim)
        self.initRewardNet(self.state_dim)

        # Architecture
        if model_type == "custom_cnn":
            if "autoencoder" in losses:
                self.model = CNNAutoEncoder(state_dim)
            elif "vae" in losses:
                self.model = CNNVAE(state_dim)

            self.model.decoder_fc = nn.Linear(self.dim_first_method, 6 * 6 * 64)

        elif model_type == "mlp":
            if "autoencoder" in losses:
                self.model = DenseAutoEncoder(input_dim=getInputDim(), state_dim=state_dim)
            elif "vae" in losses:
                self.model = DenseVAE(input_dim=getInputDim(), state_dim=state_dim)

        elif model_type == "linear":
            if "autoencoder" in losses:
                self.model = LinearAutoEncoder(input_dim=getInputDim(), state_dim=state_dim)
            else:
                raise ValueError("You must use autoencoder with linear model")

        elif model_type == "resnet":
            raise ValueError("Resnet not supported for autoencoders")

        if "triplet" in losses:
            raise ValueError("triplet not supported when splitting representation")

    def getStates(self, observations):
        """
        :param observations: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.model.getStates(observations)

    def forward(self, x):
        if "autoencoder" in self.losses:
            return self.forwardAutoencoder(x)
        elif "vae" in self.losses:
            return self.forwardVAE(x)

    def forwardVAE(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        input_shape = x.size()
        mu, logvar = self.model.encode(x)
        z = self.model.reparameterize(mu[self.first_split_indices], logvar[self.first_split_indices])
        decoded = self.model.decode(z).view(input_shape)
        return decoded, mu[self.first_split_indices], logvar[self.first_split_indices]

    def forwardAutoencoder(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        input_shape = x.size()
        encoded = self.model.encode(x)
        decoded = self.model.decode(encoded[self.first_split_indices]).view(input_shape)
        return encoded, decoded

    def inverseModel(self, state, next_state):
        """
        Predict action given current state and next state
        :param state: (th.Tensor)
        :param next_state: (th.Tensor)
        :return: probability of each action
        """
        return self.inverse_net(th.cat((state[self.second_split_indices], next_state[self.second_split_indices]), dim=1))

    def forwardModel(self, state, action):
        """
        Predict next state given current state and action
        :param state: (th.Tensor)
        :param action: (th Tensor)
        :return: (th.Tensor)
        """
        # Predict the delta between the next state and current state
        concat = th.cat((state[self.second_split_indices], encodeOneHot(action, self.action_dim)), dim=1)
        return state[self.second_split_indices] + self.forward_net(concat)

    def rewardModel(self, state, next_state):
        """
        Predict reward given current state and next state
        :param state: (th.Tensor)
        :param action: (th Tensor)
        :return: (th.Tensor)
        """
        return self.reward_net(th.cat((state, next_state), dim=1))
