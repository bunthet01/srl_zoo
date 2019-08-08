from .autoencoders import AutoEncoderTrainer
from .vae import VAETrainer
from .cvae import CVAETrainer
from .forward_inverse import BaseForwardModel, BaseInverseModel, BaseRewardModel, BaseRewardModel2, BasicTrainer, SelfSupClassfier
from .priors import SRLConvolutionalNetwork, SRLDenseNetwork, SRLLinear
from .triplet import EmbeddingNet
from .gan import GANTrainer  
from .gan_new import GanNewTrainer
from .cgan_new import CGanNewTrainer
from .cgan import CGANTrainer 
import torch
from collections import OrderedDict
try:
    # relative import: when executing as a package: python -m ...
    from ..losses.losses import forwardModelLoss, inverseModelLoss, rewardModelLoss, spclsLoss
    from .base_trainer import BaseTrainer
    from ..utils import printRed
except:
    # absolute import: when executing directly: python train.py ...
    from losses.losses import forwardModelLoss, inverseModelLoss, rewardModelLoss, spclsLoss
    from models.base_trainer import BaseTrainer
    from utils import printRed


class SRLModules(BaseForwardModel, BaseInverseModel, BaseRewardModel, BaseRewardModel2, SelfSupClassfier):
    def __init__(self, state_dim=2,cuda=False, class_dim=1,img_shape=None, action_dim=6, model_type="custom_cnn", losses=None,
                 split_dimensions=None, n_hidden_reward=16, inverse_model_type="linear", device='cpu'):
        """
        A model that can combine AE/VAE + Inverse + Forward + Reward models
        :param state_dim: (int)
        :param img_shape: (tuple or None) channels first ! 
        :param action_dim: (int)
        :param model_type: (str)
        :param losses: ([str])
        :param split_dimensions: (OrderedDict) Number of dimensions for the different losses
        :param n_hidden_reward: (int) Number of hidden units for the reward model
        :param inverse_model_type: (str) Architecture of the inverse model ('linear', 'mlp')
        """
        self.model_type = model_type
        self.losses = losses
        BaseForwardModel.__init__(self)
        BaseInverseModel.__init__(self)
        BaseRewardModel.__init__(self)
        BaseRewardModel2.__init__(self)
        SelfSupClassfier.__init__(self)
        self.state_dim = state_dim
        if img_shape is None:
            self.img_shape = (3, 224, 224)
        else:
            self.img_shape = img_shape

        # For state splitting ================= TODO UGLY
        self.split_dimensions = split_dimensions
        if self.split_dimensions != -1:
            assert len(split_dimensions) == len(losses), "Please specify as many split dimensions {} as losses {} !". \
                format(len(split_dimensions), len(losses))
            # TODO TO DELETE --------------
            n_dims = sum(split_dimensions.values())
            # Account for shared dimensions
            n_dims += list(split_dimensions.values()).count(-1)
            assert n_dims == state_dim, \
                "The sum of all splits' dimensions {} must be equal to the state dimension {}"\
                .format(sum(split_dimensions.values()), str(state_dim))
            # -------------------------------
            state_dim_dict = OrderedDict()
            prev_dim = 0
            for loss_name, dim in self.split_dimensions.items():
                if dim == -1:
                    state_dim_dict[loss_name] = prev_dim
                else:
                    state_dim_dict[loss_name] = dim
                    prev_dim = dim
        else:
            state_dim_dict = {"forward": self.state_dim, "inverse": self.state_dim, "reward": self.state_dim, "reward2": self.state_dim}

        self.initForwardNet(state_dim_dict.get("forward", self.state_dim), action_dim)
        self.initInverseNet(state_dim_dict.get("inverse", self.state_dim), action_dim, model_type=inverse_model_type)
        self.initRewardNet(state_dim_dict.get("reward", self.state_dim), n_hidden=n_hidden_reward)
        self.initRewardNet2(state_dim_dict.get("reward2", self.state_dim), n_hidden=n_hidden_reward)
        self.initSpClsmodel(state_dim_dict.get("reward2", self.state_dim), n_hidden=100)

        # Architecture
        if "autoencoder" in losses or "dae" in losses:
            self.model = AutoEncoderTrainer(state_dim=state_dim, img_shape=self.img_shape)
            self.model.build_model(model_type=model_type)
        elif "vae" in losses:
            self.model = VAETrainer(state_dim=state_dim, img_shape=self.img_shape)
            self.model.build_model(model_type=model_type)
        elif 'gan' in losses:
            self.model = GANTrainer(state_dim=state_dim, img_shape=self.img_shape)
            self.model.build_model(model_type=model_type)
        elif 'gan_new' in losses:
	        self.model = GanNewTrainer( state_dim, self.img_shape)
	        self.model.build_model(model_type=model_type)
        elif 'cgan' in losses:
            self.model = CGANTrainer(state_dim=state_dim,label_dim=class_dim, img_shape=self.img_shape, device=device)
            self.model.build_model(model_type=model_type)
        elif 'cgan_new' in losses:
            self.model = CGanNewTrainer(state_dim=state_dim,label_dim=class_dim, img_shape=self.img_shape, device=device)
            self.model.build_model(model_type=model_type)
        elif "cvae" in losses:
            self.model = CVAETrainer(state_dim=state_dim, class_dim=class_dim, img_shape=img_shape, device=device)
            self.model.build_model(model_type=model_type)
            

        # elif losses is not None and "triplet" in losses:
        #     # pretrained resnet18 with fixed weights
        #     # TODO not tested yet
        #     raise NotImplementedError
        #     self.model = EmbeddingNet(state_dim)
        else:
            # for losses not depending on specific architecture (supervised, inverse, forward..)
            self.model = BasicTrainer(state_dim=state_dim, img_shape=self.img_shape)
            self.model.build_model(model_type=model_type)  # TODO add the other model_type !!

        # elif model_type == "resnet":
        #     self.model = SRLConvolutionalNetwork(state_dim, cuda)

        # # elif: [Add new model here !]

    def forward(self, x):
        if self.model_type == 'linear' or self.model_type == 'mlp':
            x = x.contiguous()
        return self.model(x)

    def encode(self, x):
        if "triplet" in self.losses:
            return self.model(x)
        else:
            raise NotImplementedError()

    def forwardTriplets(self, anchor, positive, negative):
        """
        Overriding the forward function in the case of Triplet loss
        anchor : anchor observations (torch. Tensor)
        positive : positive observations (torch. Tensor)
        negative : negative observations (torch. Tensor)
        """
        return self.model(anchor), self.model(positive), self.model(negative)

    def add_forward_loss(self, states, actions_st, next_states, loss_manager, weight=1.0):
        next_states_pred = self.forwardModel(states, actions_st)
        forwardModelLoss(next_states_pred, next_states, weight=weight, loss_manager=loss_manager)

    def add_inverse_loss(self, states, actions_st, next_states, loss_manager, weight=2.0):
        actions_pred = self.inverseModel(states, next_states)
        inverseModelLoss(actions_pred, actions_st, weight=weight, loss_manager=loss_manager)

    def add_reward_loss(self, states, rewards_st, next_states, loss_manager, label_weights, ignore_index=-1, weight=100.0):
        # import ipdb; ipdb.set_trace()
        rewards_pred = self.rewardModel(states, next_states)
        rewardModelLoss(rewards_pred, rewards_st, weight=weight, loss_manager=loss_manager,
                        label_weights=label_weights, ignore_index=ignore_index)
    def add_reward2_loss(self, states, rewards_st, loss_manager, label_weights, ignore_index=-1, weight=100.0):
        rewards_pred = self.rewardModel2(states)
        rewardModelLoss(rewards_pred, rewards_st, weight=weight, loss_manager=loss_manager,
                        label_weights=label_weights, ignore_index=ignore_index)
    
    def add_spcls_loss(self, states, cls_gt, loss_manager, weight=100.0):
        cls_pred = self.classifier(states)
        spclsLoss(cls_pred, cls_gt, weight=weight, loss_manager=loss_manager)
