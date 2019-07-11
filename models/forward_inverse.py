from __future__ import print_function, division, absolute_import

import torch

from .base_models import *
from .base_trainer import BaseTrainer


class BaseForwardModel(BaseModelSRL):
    def __init__(self):
        self.action_dim = None
        self.forward_net = None
        super(BaseForwardModel, self).__init__()

    def initForwardNet(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.forward_net = nn.Linear(state_dim + action_dim, state_dim)

    def forward(self, x):
        raise NotImplementedError()

    def forwardModel(self, state, action):
        """
        Predict next state given current state and action
        :param state: (torch.Tensor)
        :param action: (torch.Tensor)
        :return: (torch.Tensor)
        """
        # Predict the delta between the next state and current state
        # by taking as input concatenation of state & action over the 2nd dimension
        concat = torch.cat((state, encodeOneHot(action, self.action_dim)), dim=1)
        return state + self.forward_net(concat)


class BaseInverseModel(BaseModelSRL):
    def __init__(self):
        self.inverse_net = None
        super(BaseInverseModel, self).__init__()

    def initInverseNet(self, state_dim, action_dim, n_hidden=128, model_type="linear"):
        """
        :param state_dim: (torch.Tensor)
        :param action_dim: (int)
        :param n_hidden: (int)
        :param model_type: (str)
        :return: (torch.Tensor)
        """
        if model_type == "linear":
            self.inverse_net = nn.Linear(state_dim, action_dim)
        elif model_type == "mlp":
            self.inverse_net = nn.Sequential(nn.Linear(state_dim * 2, n_hidden),
                                             nn.ReLU(),
                                             nn.Linear(n_hidden, n_hidden),
                                             nn.ReLU(),
                                             nn.Linear(n_hidden, action_dim)
                                             )
        else:
            raise ValueError("Unknown model_type for inverse model: {}".format(model_type))

    def forward(self, x):
        raise NotImplementedError()

    def inverseModel(self, state, next_state):
        """
        Predict action given current state and next state
        :param state: (torch.Tensor)
        :param next_state: (torch.Tensor)
        :return: probability of each action
        """
        # input: concatenation of state & next state over the 2nd dimension
        return self.inverse_net(next_state-state)


class BaseRewardModel(BaseModelSRL):
    def __init__(self):
        self.reward_net = None
        super(BaseRewardModel, self).__init__()

    def initRewardNet(self, state_dim, n_rewards=2, n_hidden=16):
        self.reward_net = nn.Sequential(nn.Linear(2*state_dim, n_hidden),
                                        nn.LeakyReLU(negative_slope=0.1),
                                        nn.Linear(n_hidden, n_hidden),
                                        nn.LeakyReLU(negative_slope=0.1),
                                        nn.Linear(n_hidden, n_rewards))
    def forward(self, x):
        raise NotImplementedError()

    def rewardModel(self, state, next_state):
        """
        Predict reward given current state and next state
        :param state: (torch.Tensor)
        :param next_state: (torch.Tensor)
        :return: (torch.Tensor)
        """
        return self.reward_net(torch.cat((state, next_state), dim=1))

        


class BaseRewardModel2(BaseModelSRL):
    def __init__(self):
        self.reward_net2 = None
        super(BaseRewardModel2, self).__init__()

    def initRewardNet2(self, state_dim, n_rewards=2, n_hidden=16):
        self.reward_net2 = nn.Sequential(nn.Linear(state_dim, n_hidden),
                                        nn.LeakyReLU(negative_slope=0.1),
                                        nn.Linear(n_hidden, n_hidden),
                                        nn.LeakyReLU(negative_slope=0.1),
                                        nn.Linear(n_hidden, n_rewards))
    def forward(self, x):
        raise NotImplementedError()

    def rewardModel2(self, state):
        """
        Predict reward given current state and next state
        :param state: (torch.Tensor)
        :return: (torch.Tensor)
        """
        return self.reward_net2(state)
        # another idea
        # concat = torch.cat((state, encodeOneHot(action, self.action_dim)), dim=1)
        # return self.reward_net(concat)

class SelfSupClassfier(BaseModelSRL):
    def __init__(self, num_classes=100):
        super(SelfSupClassfier, self).__init__()
        self.classifier = None
        self.num_classes = num_classes
        
    def initSpClsmodel(self, state_dim, n_hidden=100):
        self.state_dim = state_dim
        self.classifier = nn.Sequential(nn.Linear(self.state_dim, n_hidden),
                                        nn.BatchNorm1d(n_hidden),
                                        nn.LeakyReLU(negative_slope=0.1),
                                        nn.Linear(n_hidden, n_hidden),
                                        nn.BatchNorm1d(n_hidden),
                                        nn.LeakyReLU(negative_slope=0.1),
                                        nn.Linear(n_hidden, self.num_classes)
                                        )
    def forward(self, x):
        raise NotImplementedError()

class BasicTrainer(BaseTrainer):
    """
    Define basic trainer for Inverse, Forward model. 

    """

    def __init__(self, state_dim=2, img_shape=(3, 224, 224)):
        super().__init__()
        self.state_dim = state_dim
        self.img_shape = img_shape

    def build_model(self, model_type=None):

        self.model = CustomCNN(self.state_dim, self.img_shape)

    def train_on_batch(self, obs, next_obs, optimizer, loss_manager, valid_mode=False, device=torch.device('cpu')):
        """
        :param obs, next_obs (useless here, since the loss have already been accumulated outside)
        :param optimizer: pytorch optimizer
        :param loss_manager: collect loss tensors and loss history
        :param valid_model (bool) validation mode (or training mode)
        return loss: (scalar)
        """
        # Define the training mechanism here
        # ------------- It's mandatory to update loss/model weights by calling -----------
        loss = self.update_nn_weights(optimizer, loss_manager, valid_mode=valid_mode)
        return loss

    def forward(self, x):
        return self.model(x)
