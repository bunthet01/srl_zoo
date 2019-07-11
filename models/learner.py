
from __future__ import print_function, division, absolute_import

import os
import json
import sys
import time
from collections import defaultdict, OrderedDict
from pprint import pprint


import numpy as np
import torch
from torchvision.utils import make_grid
import torch.utils.data
from sklearn.utils import shuffle as sk_shuffle
from tqdm import tqdm

from losses.losses import LossManager, autoEncoderLoss, roboticPriorsLoss, tripletLoss, rewardModelLoss, \
    rewardPriorLoss, forwardModelLoss, inverseModelLoss, episodePriorLoss, l1Loss, l2Loss, kullbackLeiblerLoss, \
    kullbackLeiblerLossCCI, generationLoss, ganNonSaturateLoss
from losses.utils import findPriorsPairs
from pipeline import NAN_ERROR
from plotting.representation_plot import plotRepresentation, plotImage, printGTC
from preprocessing.data_loader import convertScalerToVectorAction, RobotEnvDataset
from preprocessing.utils import deNormalize
from utils import printRed, detachToNumpy, printYellow
from .modules import SRLModules
from .priors import Discriminator as PriorDiscriminator


MAX_BATCH_SIZE_GPU = 256  # For plotting, max batch_size before having memory issues
EPOCH_FLAG = 1  # Plot every 1 epoch
ITER_FLAG = 1  # Print loss every 10 iterations
N_WORKERS = 10

# The following variables are defined using arguments of the main script train.py
# SAVE_PLOTS = True
BATCH_SIZE = 256
N_EPOCHS = 1
VALIDATION_SIZE = 0.2  # 20% of training data for validation
# Experimental: episode independent prior
# Whether to do Uniform (default) or balanced sampling
BALANCED_SAMPLING = False


class BaseLearner(object):
    """
    Base class for a method that learn a state representation
    from observations
    :param state_dim: (int)
    :param batch_size: (int)
    :param seed: (int)
    :param cuda: (int) (default -1, CPU) equi to CUDA_VISIBLE_DEVICES
    """

    def __init__(self, state_dim, class_dim, batch_size, seed=1, cuda=-1, use_cvae=False):
        super(BaseLearner, self).__init__()
        self.state_dim = state_dim
        self.class_dim = class_dim
        self.batch_size = batch_size
        self.module = None
        self.seed = seed
        self.use_dae = False
        self.use_cvae = use_cvae
        assert not self.use_dae, "Not implemented error."
        # Seed the random generator
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda >= 0:  # This will increase the training time by 5-10%.
            # Make CuDNN Determinist
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed(seed)

        self.device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() and (cuda >= 0) else "cpu")

    def _predFn(self, observations, action, use_cvae):
        """
        Predict states in test mode given observations

        :param observations: (torch.Tensor)
        :return: (np.ndarray)
        """
        # Move the tensor back to the cpu
        if use_cvae:
            return detachToNumpy(self.module.model(observations, action))
        else:
            return detachToNumpy(self.module.model(observations))

    def predStatesWithDataLoader(self, data_loader):
        """
        Predict states using minibatches to avoid memory issues
        :param data_loader: (DataLoader object)
        :return: (np.ndarray)
        """
        predictions = []
        for batch in data_loader:
            obs = batch[0].to(self.device)
            action = batch[1].to(self.device)
            predictions.append(self._predFn(obs, action, self.use_cvae))

        return np.concatenate(predictions, axis=0)

    def learn(self, *args, **kwargs):
        """
        Function called to learn a state representation
        it returns the learned states for the given observations
        """
        raise NotImplementedError("Learn method not implemented")

    @staticmethod
    def saveStates(states, images_path, rewards, log_folder, name=""):
        """
        Save learned states to json and npz files

        :param states: (np.ndarray)
        :param images_path: ([str])
        :param rewards: (rewards)
        :param log_folder: (str)
        :param name: (str)
        """
        print("Saving image path to state representation (image_to_state{}.json)".format(name))

        image_to_state = {path: list(map(str, state)) for path, state in zip(images_path, states)}

        with open("{}/image_to_state{}.json".format(log_folder, name), 'w') as f:
            json.dump(image_to_state, f, sort_keys=True)

        print("Saving states and rewards (states_rewards{}.npz)".format(name))

        states_rewards = {'states': states, 'rewards': rewards}
        np.savez('{}/states_rewards{}.npz'.format(log_folder, name), **states_rewards)


def infinite_dataloader(dataloader):
    while True:
        for X in dataloader:
            yield X


class SRL4robotics(BaseLearner):
    """
    Main Class for training a SRL model

    :param state_dim: (int)
    :param model_type: (str) one of "resnet", "mlp" or "custom_cnn"
    :param inverse_model_type: (str) one of "linear" or "mlp"
    :param log_folder: (str)
    :param seed: (int)
    :param learning_rate: (float)
    :param learning_rate_gan: tuple of float (lr_D, lr_G)
    :param l1_reg: (float) weight for l1 regularization
    :param l2_reg: (float) weight for l2 regularization
    :param cuda: (int) (default -1, CPU) equi to CUDA_VISIBLE_DEVICES
    :param multi_view: (bool)
    :param losses: ([str])
    :param losses_weights_dict: (OrderedDict)
    :param n_actions: (int)
    :param beta: (float) for beta-vae
    :param split_dimensions:
    :param path_to_dae: (str) path to pre-trained DAE when using perceptual loss
    :param state_dim_dae: (int)
    :param occlusion_percentage: (float) max percentage of occlusion when using DAE
    :param pretrained_weights_path: SRL pretrained model weights path (default: None)
    """

    def __init__(self, state_dim, class_dim, img_shape=None, model_type="resnet", inverse_model_type="linear", log_folder="logs/default",
                 seed=1, learning_rate=0.001, learning_rate_gan=(0.001, 0.001), l1_reg=0.0, l2_reg=0.0, cuda=-1,
                 multi_view=False, losses=None, losses_weights_dict=None, n_actions=6, beta=1,
                 split_dimensions=-1, path_to_dae=None, state_dim_dae=200, occlusion_percentage=None, pretrained_weights_path=None):

        super(SRL4robotics, self).__init__(state_dim, class_dim, BATCH_SIZE, seed, cuda, use_cvae="cvae" in losses)

        self.multi_view = multi_view
        self.losses = losses
        self.dim_action = n_actions
        self.beta = beta
        self.denoiser = None
        self.img_shape = img_shape
        self.model_type = model_type
        self.pretrained_weights_path = pretrained_weights_path
        if model_type in ["linear", "mlp", "resnet", "custom_cnn", 'gan', 'unet'] \
                or "autoencoder" in losses or "vae" in losses:
            self.use_forward_loss = "forward" in losses
            self.use_inverse_loss = "inverse" in losses
            self.use_reward_loss = "reward" in losses
            self.no_priors = "priors" not in losses
            self.episode_prior = "episode-prior" in losses
            self.reward_prior = "reward-prior" in losses
            self.use_autoencoder = "autoencoder" in losses
            self.use_vae = "vae" in losses
            self.use_cvae = "cvae" in losses
            self.use_triplets = "triplet" in self.losses
            # self.perceptual_similarity_loss = "perceptual" in self.losses
            self.use_dae = "dae" in self.losses
            self.path_to_dae = path_to_dae

            if isinstance(split_dimensions, OrderedDict) and sum(split_dimensions.values()) > 0:
                printYellow("Using splitted representation")
                self.use_split = True
                self.split_dimensions = split_dimensions  # TODO UGLY!
            else:
                self.use_split = False
            self.module = SRLModules(state_dim=self.state_dim, class_dim = self.class_dim, img_shape=self.img_shape, action_dim=self.dim_action, model_type=model_type,
                                     losses=losses, split_dimensions=split_dimensions, inverse_model_type=inverse_model_type)
        else:
            raise ValueError("Unknown model: {}".format(model_type))

        print("Using {} model".format(model_type))

        self.cuda = cuda
        self.device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() and (cuda >= 0) else "cpu")

        if self.episode_prior:
            self.prior_discriminator = PriorDiscriminator(2 * self.state_dim).to(self.device)

        self.module = self.module.to(self.device)

        if self.model_type != 'gan':
            learnable_params = [param for param in self.module.parameters() if param.requires_grad]
            # learnable_params = list(self.module.parameters()) ## NEW [TODO]

            if self.episode_prior:
                learnable_params += [p for p in self.prior_discriminator.parameters()]
            self.optimizer = torch.optim.Adam(learnable_params, lr=learning_rate)
        else:
            assert not self.episode_prior, "NotImplementedError"
            self.optimizer_D = torch.optim.Adam(
                self.module.model.discriminator.parameters(), lr=learning_rate_gan[0], betas=(0.5, 0.9))
            self.optimizer_G = torch.optim.Adam(
                self.module.model.generator.parameters(), lr=learning_rate_gan[1], betas=(0.5, 0.9))
            combined_E_params = [param for param in self.module.model.encoder.parameters()]
            combined_E_params += [param for param in self.module.model.generator.parameters()]
            combined_E_params += [param for param in self.module.forward_net.parameters()]
            combined_E_params += [param for param in self.module.inverse_net.parameters()]
            combined_E_params += [param for param in self.module.reward_net.parameters()]
            self.optimizer = torch.optim.Adam(combined_E_params, lr=learning_rate, betas=(0.5, 0.9))
            # [TODO: check learnable parameters]
        self.log_folder = log_folder

        # Default weights that are updated with the weights passed to the script
        self.losses_weights_dict = {"forward": 1.0, "inverse": 2.0, "reward": 1.0, "priors": 1.0,
                                    "episode-prior": 1.0, "reward-prior": 10, "triplet": 1.0,
                                    "autoencoder": 1.0, "vae": 1.0, "cvae": 1.0, "perceptual": 1e-6, "dae": 1.0,
                                    'l1_reg': l1_reg, "l2_reg": l2_reg, 'random': 1.0}
        self.occlusion_percentage = occlusion_percentage
        self.state_dim_dae = state_dim_dae

        if losses_weights_dict is not None:
            self.losses_weights_dict.update(losses_weights_dict)

        # if self.use_dae and self.occlusion_percentage is not None:
        #     print("Using a maximum occlusion surface of {}".format(str(self.occlusion_percentage)))

    @staticmethod
    def loadSavedModel(log_folder, valid_models, cuda=True, img_shape=None):
        """
        Load a saved SRL model

        :param log_folder: (str)
        :param valid_models: ([str])
        :param cuda: (bool)
        :return: (SRL4robotics object, OrderedDict)
        """
        # Sanity checks
        assert os.path.exists(log_folder), "Error: folder '{}' does not exist".format(log_folder)
        assert os.path.exists(log_folder + "exp_config.json"), \
            "Error: could not find 'exp_config.json' in '{}'".format(log_folder)
        assert os.path.exists(log_folder + "srl_model.pth"), \
            "Error: could not find 'srl_model.pth' in '{}'".format(log_folder)

        with open(log_folder + 'exp_config.json', 'r') as f:
            # IMPORTANT: keep the order for the losses
            # so the json is loaded as an OrderedDict
            exp_config = json.load(f, object_pairs_hook=OrderedDict)

        state_dim = exp_config['state-dim']
        class_dim = exp_config['class_dim']
        losses = exp_config['losses']
        n_actions = exp_config['n_actions']
        model_type = exp_config['model-type']
        multi_view = exp_config.get('multi-view', False)
        split_dimensions = exp_config.get('split-dimensions', -1)
        model_path = log_folder + 'srl_model.pth'
        inverse_model_type = exp_config.get('inverse-model-type', 'linear')
        occlusion_percentage = exp_config.get('occlusion-percentage', 0)

        difference = set(losses).symmetric_difference(valid_models)

        # HACK: TODO include GAN to the losses and add it to valid_models of ./srl_zoo/evaluation/enjoy_latent.py
        # assert set(losses).intersection(valid_models) != set(), "Error: Not supported losses " + ", ".join(difference)

        srl_model = SRL4robotics(state_dim, class_dim, img_shape=img_shape, model_type=model_type, cuda=cuda, multi_view=multi_view,
                                 losses=losses, n_actions=n_actions, split_dimensions=split_dimensions,
                                 inverse_model_type=inverse_model_type, occlusion_percentage=occlusion_percentage)
        srl_model.module.load_state_dict(torch.load(model_path))

        return srl_model, exp_config

    def learn(self, images_path, actions, rewards, episode_starts,
              figdir=None,
              monitor_mode='loss',
              pretrained_weights_path=None,
              ground_truth=None,
              relative_positions=None,
              target_positions=None
              ):
        """
        Learn a state representation
        :param images_path: (numpy 1D array)
        :param actions: (np.ndarray)
        :param rewards: (numpy 1D array)
        :param episode_starts: (numpy 1D array) boolean array
                                the ith index is True if one episode starts at this frame
        :param figdir: directory path, save figures to the folder.
        :param monitor_mode: options are ['loss', 'pbar']
        :param ground_truth (dict) for the plots and GTC (ground truth correlation)
        :param relative_positions: (np.ndarray) for the plots and GTC (ground truth correlation)
        :param target_positions: (np.ndarray) for the plots and GTC (ground truth correlation)

        :return: (np.ndarray) the learned states for the given observations
        """

        print("\nYour are using the following weights for the losses:")
        pprint(self.losses_weights_dict)
        assert monitor_mode in ['loss', 'pbar'], "monitor should be either 'loss' or 'prgressbar'"
        # PREPARE DATA -------------------------------------------------------------------------------------------------
        # here, we organize the data into minibatches
        # and find pairs for the respective loss terms (for robotics priors only)
        if figdir is not None:
            figdir_repr = os.path.join(figdir, "state_repr")  # state representation scatter plot
            figdir_recon = os.path.join(figdir, "img_recon")  # image reconstruction folder
            os.makedirs(figdir_repr, exist_ok=True)
            os.makedirs(figdir_recon, exist_ok=True)

        data_loader_params = {'batch_size': self.batch_size,
                              'shuffle': True,
                              'num_workers': N_WORKERS,
                              #   'drop_last': False,
                              'pin_memory': False
                              }
        data_loader_params_test = {'batch_size': 128,
                                   'shuffle': False,
                                   'num_workers': N_WORKERS,
                                   #   'drop_last': False,
                                   'pin_memory': False
                                   }
        sample_indices = np.arange(len(images_path))
        # Shuffle datasets
        # sample_indices, images_path, actions, rewards, episode_starts = sk_shuffle(sample_indices, images_path, actions, rewards, episode_starts, random_state=0)
        sample_indices = sk_shuffle(sample_indices, random_state=0)
        valid_size = np.round(VALIDATION_SIZE * len(images_path)).astype(np.int64)

        # indices_train, imgspath_train, act_train, rew_train, epis_train = sample_indices[:-valid_size], images_path[:-valid_size], actions[:-valid_size], rewards[:-valid_size], episode_starts[:-valid_size]
        # indices_val, imgspath_val, act_val, rew_val, epis_val           = sample_indices[-valid_size:], images_path[-valid_size:], actions[-valid_size:], rewards[-valid_size:], episode_starts[-valid_size:]
        indices_train, indices_val = sample_indices[:-valid_size], sample_indices[-valid_size:]
        train_set = RobotEnvDataset(indices_train, images_path, actions, rewards, episode_starts,
                                    is_training=True, img_shape=self.img_shape, multi_view=self.multi_view,
                                    use_triplets=self.use_triplets, apply_occlusion=self.use_dae,
                                    occlusion_percentage=self.occlusion_percentage, dtype=np.float32)
        valid_set = RobotEnvDataset(indices_val, images_path, actions, rewards, episode_starts,
                                    is_training=True, img_shape=self.img_shape, multi_view=self.multi_view,
                                    use_triplets=self.use_triplets, apply_occlusion=self.use_dae,
                                    occlusion_percentage=self.occlusion_percentage, dtype=np.float32)
        test_set = RobotEnvDataset(np.arange(len(images_path)), images_path, actions, rewards, episode_starts,
                                   is_training=False, img_shape=self.img_shape, multi_view=self.multi_view,
                                   use_triplets=self.use_triplets, apply_occlusion=self.use_dae,
                                   occlusion_percentage=self.occlusion_percentage, dtype=np.float32)
        dataloader_train = torch.utils.data.DataLoader(train_set, **data_loader_params)
        dataloader_valid = torch.utils.data.DataLoader(valid_set, **data_loader_params)
        dataloader_test = torch.utils.data.DataLoader(test_set, **data_loader_params_test)
        # ------- Load ground truth states/ target positions (for plots and GTC) -------

        # ========================= Print some info =========================
        # Stats about actions
        action_set = set(actions)
        n_actions = int(np.max(actions) + 1)
        print("{} unique actions / {} actions".format(len(action_set), n_actions))
        n_pairs_per_action = np.zeros(n_actions, dtype=np.int64)
        n_obs_per_action = np.zeros(n_actions, dtype=np.int64)
        for i in range(n_actions):
            n_obs_per_action[i] = np.sum(actions == i)
        print("Number of observations per action")
        print(n_obs_per_action)
        print("Train: {} minibatches, {} samples".format(len(dataloader_train), len(train_set)))
        print("Valid: {} minibatches, {} samples".format(len(dataloader_valid), len(valid_set)))
        # =======================================================================
        # Load pretrained model weights
        if self.pretrained_weights_path is not None:
            assert os.path.exists(self.pretrained_weights_path), "Model weigths file: {} DOESN'T exist.".format(
                self.pretrained_weights_path)
            print("Loading pretrained model weights from path: {}...".format(self.pretrained_weights_path))
            self.module.load_state_dict(torch.load(self.pretrained_weights_path))

        # dissimilar_pairs, same_actions_pairs = None, None
        # if not self.no_priors:
        #     dissimilar_pairs, same_actions_pairs = findPriorsPairs(self.batch_size, minibatchlist, actions, rewards,
        #                                                            n_actions, n_pairs_per_action)
        # if self.use_vae and self.perceptual_similarity_loss and self.path_to_dae is not None:
        #     self.denoiser = SRLModules(state_dim=self.state_dim_dae, img_shape=self.img_shape, action_dim=self.dim_action,
        #                                model_type="custom_cnn",
        #                                losses=["dae"])
        #     self.denoiser.load_state_dict(torch.load(self.path_to_dae))
        #     self.denoiser.eval()
        #     self.denoiser = self.denoiser.to(self.device)
        #     for param in self.denoiser.parameters():
        #         param.requires_grad = False
        # if self.episode_prior:
        #     idx_to_episode = {idx: episode_idx for idx, episode_idx in enumerate(np.cumsum(episode_starts))}
        #     minibatch_episodes = [[idx_to_episode[i] for i in minibatch] for minibatch in minibatchlist]

        # TRAINING -----------------------------------------------------------------------------------------------------
        loss_history = defaultdict(list)
        loss_manager = LossManager(self.module, loss_history)
        if self.model_type == 'gan':
            loss_history_D = defaultdict(list)
            loss_history_G = defaultdict(list)
            loss_manager_D = LossManager(self.module.model.discriminator, loss_history_D)
            loss_manager_G = LossManager(self.module.model.generator, loss_history_G)
            # define the labels here so that we can reuse them (save time)
            label_valid = torch.ones((self.batch_size, 1)).to(self.device)
            label_fake = torch.zeros((self.batch_size, 1)).to(self.device)
        best_error = np.inf
        best_model_path = "{}/srl_model.pth".format(self.log_folder)

        # Random features, we don't need to train a model
        if len(self.losses) == 1 and self.losses[0] == 'random':
            global N_EPOCHS
            N_EPOCHS = 0
            printYellow("Skipping training because using random features")
            torch.save(self.module.state_dict(), best_model_path)

        for epoch in range(N_EPOCHS):
            for valid_mode, dataloader in enumerate([dataloader_train, dataloader_valid]):
                if monitor_mode == 'pbar':
                    pbar = tqdm(total=len(dataloader))
                if self.model_type == 'gan':
                    # GAN's training requires multi-optimizers, thus multiple loss_manager/epoch_loss/val_loss, etc.
                    epoch_loss_D, epoch_batches_D = 0, 0
                    epoch_loss_G, epoch_batches_G = 0, 0
                    d_acc, g_acc = 0, 0
                epoch_loss, epoch_batches = 0, 0
                n_batch_per_epoch = len(dataloader)

                start_time = time.time()
                if valid_mode:
                    self.module.eval()
                    # with torch.no_grad():
                    self.prev_grad_mode = torch.is_grad_enabled()
                    torch.set_grad_enabled(False)
                    # torch._C.set_grad_enabled(False) ??
                else:
                    self.module.train()
                dataloader = infinite_dataloader(dataloader)  # iter(dataloader)
                for iter_ind in range(n_batch_per_epoch):
                    (sample_idx, obs, next_obs, action, next_action, reward, noisy_obs, next_noisy_obs) = next(dataloader)
                    obs, next_obs = obs.to(self.device), next_obs.to(self.device)

                    # if self.use_dae:
                    #     noisy_obs = noisy_obs.to(self.device)
                    #     next_noisy_obs = next_noisy_obs.to(self.device)

                    if self.model_type != "gan":
                        # It's extremely important to release loss tensor, otherwise it will cause 'memory leak".
                        self.optimizer.zero_grad()
                        loss_manager.resetLosses()

                    # Predict states given observations as in Time Contrastive Network (Triplet Loss) [Sermanet et al.]
                    # if self.use_triplets:
                    #     states, positive_states, negative_states = self.module.forwardTriplets(obs[:, :3:, :, :],
                    #                                                                         obs[:, 3:6, :, :],
                    #                                                                         obs[:, 6:, :, :])

                    #     next_states, next_positive_states, next_negative_states = self.module.forwardTriplets(
                    #         next_obs[:, :3:, :, :],
                    #         next_obs[:, 3:6, :, :],
                    #         next_obs[:, 6:, :, :])
                    # elif self.use_dae:
                    #     (states, reconstruct_obs), (next_states, decoded_next_obs) = \
                    #         self.module(noisy_obs), self.module(next_noisy_obs)

                    # elif self.use_vae:
                    #     states, next_states = self.module(obs), self.module(next_obs)

                    #     if self.perceptual_similarity_loss:
                    #         # Predictions for the perceptual similarity loss as in DARLA
                    #         # https://arxiv.org/pdf/1707.08475.pdf
                    #         (states_denoiser, decoded_obs_denoiser), (next_states_denoiser, decoded_next_obs_denoiser) = \
                    #             self.denoiser(obs), self.denoiser(next_obs)

                    #         (states_denoiser_predicted, decoded_obs_denoiser_predicted) = self.denoiser(reconstruct_obs)
                    #         (next_states_denoiser_predicted, decoded_next_obs_denoiser_predicted) = self.denoiser(next_decoded_obs)
                    # else:
                    #     states, next_states = self.module(obs), self.module(next_obs)

                    # L1 regularization
                    if self.losses_weights_dict['l1_reg'] > 0:
                        l1Loss(loss_manager.reg_params,
                               self.losses_weights_dict['l1_reg'], loss_manager)
                    if self.losses_weights_dict['l2_reg'] > 0:
                        l2Loss(loss_manager.reg_params,
                               self.losses_weights_dict['l2_reg'], loss_manager)
                    # if not self.no_priors:
                    #     roboticPriorsLoss(states, next_states, minibatch_idx=minibatch_idx,
                    #                     dissimilar_pairs=dissimilar_pairs, same_actions_pairs=same_actions_pairs,
                    #                     weight=self.losses_weights_dict['priors'], loss_manager=loss_manager)
                    # Actions associated to the observations
                    if self.use_forward_loss or self.use_inverse_loss or self.use_reward_loss:
                        states, next_states = self.module(obs), self.module(next_obs)
                        actions_st = action.view(-1, 1).to(self.device)

                    if not self.use_split:
                        if self.use_forward_loss:
                            self.module.add_forward_loss(states, actions_st, next_states, loss_manager)
                        if self.use_inverse_loss:
                            self.module.add_inverse_loss(states, actions_st, next_states, loss_manager)
                        if self.use_reward_loss:
                            rewards_st = np.array(reward).copy()
                            # Removing negative reward
                            # rewards_st[rewards_st == -1] = 0
                            rewards_st = torch.from_numpy(rewards_st.astype(int)).to(self.device)
                            label_weights = torch.tensor([0.0, 1000.0]).to(self.device)
                            self.module.add_reward_loss(states, rewards_st, next_states, loss_manager,
                                                        label_weights=label_weights, ignore_index=-1)
                    else:  # TODO UGLY
                        split_dim_list = [a for a in list(self.split_dimensions.values()) if a != -1]
                        states_split_list = torch.split(states, split_dim_list, dim=-1)
                        next_states_split_list = torch.split(next_states, split_dim_list, dim=-1)
                        state_index = OrderedDict()
                        count_index = 0
                        prev_index = 0
                        for loss_name, state_split_dim in self.split_dimensions.items():
                            if state_split_dim == -1:
                                state_index[loss_name] = prev_index
                            else:
                                state_index[loss_name] = count_index
                                prev_index = count_index
                                count_index += 1
                        if self.use_forward_loss:
                            self.module.add_forward_loss(states_split_list[state_index["forward"]],
                                                         actions_st, next_states_split_list[state_index["forward"]], loss_manager)
                        if self.use_inverse_loss:
                            self.module.add_inverse_loss(states_split_list[state_index["inverse"]],
                                                         actions_st, next_states_split_list[state_index["inverse"]], loss_manager)
                        if self.use_reward_loss:
                            rewards_st = np.array(reward).copy()
                            # Removing negative reward
                            # rewards_st[rewards_st == -1] = 0
                            rewards_st = torch.from_numpy(rewards_st.astype(int)).to(self.device)
                            label_weights = torch.tensor([0.0, 100.0]).to(self.device)
                            self.module.add_reward_loss(states_split_list[state_index["reward"]],
                                                        rewards_st, 
                                                        next_states_split_list[state_index["reward"]], 
                                                        loss_manager,
                                                        label_weights=label_weights, 
                                                        ignore_index=-1)

                    # if self.use_vae:
                    #     if self.perceptual_similarity_loss:
                    #         perceptualSimilarityLoss(states_denoiser, states_denoiser_predicted, next_states_denoiser,
                    #                                 next_states_denoiser_predicted,
                    #                                 weight=self.losses_weights_dict['perceptual'],
                    #                                 loss_manager=loss_manager)
                    #     else:
                    #         generationLoss(reconstruct_obs, next_decoded_obs, obs, next_obs,
                    #                     weight=self.losses_weights_dict['vae'], loss_manager=loss_manager)
                    # if self.reward_prior:
                    #     rewards_st = reward
                    #     rewards_st = torch.from_numpy(rewards_st).float().view(-1, 1).to(self.device)
                    #     rewardPriorLoss(states, rewards_st, weight=self.losses_weights_dict['reward-prior'],
                    #                     loss_manager=loss_manager)
                    # if self.episode_prior:
                    #     episodePriorLoss(minibatch_idx, minibatch_episodes, states, self.prior_discriminator,
                    #                     BALANCED_SAMPLING, weight=self.losses_weights_dict['episode-prior'],
                    #                     loss_manager=loss_manager)
                    # if self.use_triplets:
                    #     tripletLoss(states, positive_states, negative_states, weight=self.losses_weights_dict['triplet'],
                    #                 loss_manager=loss_manager, alpha=0.2)
                    if self.model_type == 'gan':
                        # GAN's training requires multi-optimizers.
                        if not valid_mode:
                            # === Train the Discriminator ===
                            D_steps = 3 if (d_acc < 0.8) else 1
                            G_steps = 3 if (g_acc < 0.2) else 1
                            acc_cum = 0
                            for _ in range(D_steps):
                                self.optimizer_D.zero_grad()
                                loss_manager_D.resetLosses()
                                (sample_idx, obs, next_obs, action, reward, noisy_obs, next_noisy_obs) = next(dataloader)
                                obs = obs.to(self.device)
                                # re-define the length label_valid/label_fake, because obs.size(0) changes
                                d_loss, d_acc = self.module.model.train_on_batch_D(obs, label_valid[:obs.size(0)], label_fake[:obs.size(
                                    0)], self.optimizer_D, loss_manager_D, valid_mode=valid_mode, device=self.device)
                                epoch_loss_D += d_loss
                                epoch_batches_D += 1
                                acc_cum += d_acc
                            d_acc = acc_cum / D_steps

                            # === Train the Generator ===
                            acc_cum = 0
                            for _ in range(G_steps):
                                self.optimizer_G.zero_grad()
                                loss_manager_G.resetLosses()
                                (sample_idx, obs, next_obs, action, reward, noisy_obs, next_noisy_obs) = next(dataloader)
                                obs = obs.to(self.device)
                                # re-define the length label_valid, because obs.size(0) changes
                                g_loss, g_acc = self.module.model.train_on_batch_G(obs, label_valid[:obs.size(
                                    0)], self.optimizer_G, loss_manager_G, valid_mode=valid_mode, device=self.device)
                                epoch_loss_G += g_loss
                                epoch_batches_G += 1
                                acc_cum += g_acc
                            g_acc = acc_cum / G_steps
                        # === Train the Encoder and the other components (e.g. forward/inverse/reward model) ===
                        E_steps = 10 if not valid_mode else 1
                        for _ in range(E_steps):
                            self.optimizer.zero_grad()
                            loss_manager.resetLosses()
                            (sample_idx, obs, next_obs, action, reward, noisy_obs, next_noisy_obs) = next(dataloader)
                            obs, next_obs = obs.to(self.device), next_obs.to(self.device)
                            e_loss = self.module.model.train_on_batch_E(
                                obs, next_obs, self.optimizer, loss_manager, valid_mode=valid_mode, device=self.device)
                            epoch_loss += e_loss
                            epoch_batches += 1
                        if not valid_mode:
                            train_loss_D = epoch_loss_D / float(epoch_batches_D)
                            train_loss_G = epoch_loss_G / float(epoch_batches_G)
                            train_loss = epoch_loss / float(epoch_batches)
                        else:
                            val_loss = epoch_loss / float(epoch_batches)
                        # Custom/Optional plots
                        if iter_ind % 20 == 0 and not valid_mode:
                            reconstruct_obs = self.module.model.reconstruct(obs)
                            # , normalize=True, range=(0,1)
                            images = make_grid([obs[0], reconstruct_obs[0], obs[1], reconstruct_obs[1]], nrow=2)
                            plotImage(deNormalize(detachToNumpy(images)), mode='cv2',
                                      save2dir=figdir_recon, index=(epoch*n_batch_per_epoch+iter_ind))
                        if iter_ind % 300 == 0 and iter_ind > 0 and not valid_mode:
                            # [TODO: with no_grad() and model.eval() ???]
                            plotRepresentation(self.predStatesWithDataLoader(dataloader_test), rewards,
                                               name="Learned State Representation (Training Data)",
                                               fit_pca=False,
                                               path=os.path.join(figdir_repr, "Iter_{}.png".format(
                                                   epoch*n_batch_per_epoch+iter_ind)),
                                               verbose=False)
                        # Custom save losses
                        if not valid_mode:
                            with open(os.path.join(self.log_folder, "loss_history.csv"), "a") as file:
                                np.savetxt(file, np.array([train_loss, train_loss_D, train_loss_G,
                                                           d_acc, g_acc]).reshape(1, -1))

                        if monitor_mode == 'loss':

                            if iter_ind % ITER_FLAG == 0 or (iter_ind == n_batch_per_epoch-1):
                                if not valid_mode:
                                    print("\rEpoch {:3}/{}, {:.2%}, E_loss: {:.2f} D_loss: {:.4f} acc: {:.1%} G_loss: {:.4f} acc: {:.1%}| (elapsed time: {:.2f}s)".format(
                                        epoch + 1, N_EPOCHS, (iter_ind+1)/n_batch_per_epoch, train_loss, train_loss_D, d_acc, train_loss_G, g_acc, time.time() - start_time), end="")
                                else:
                                    val_loss_str = "{:.2f}*".format(
                                        val_loss) if val_loss < best_error else "{:.2f}".format(val_loss)
                                    print("\r-------(valid): {:.2%}, E_loss: {} | (elapsed time: {:.2f}s)".format(
                                        (iter_ind+1)/n_batch_per_epoch, val_loss_str, time.time() - start_time), end="")
                        elif monitor_mode == 'pbar':
                            pbar.update(1)

                    else:  # ============= Main update ====================

                        # Compute weighted average of losses of encoder part (including 'forward'/'inverse'/'reward' models)
                        if self.use_cvae:
                            action_cvae = convertScalerToVectorAction(action).to(self.device)
                            next_action_cvae = convertScalerToVectorAction(next_action).to(self.device)

                            loss = self.module.model.train_on_batch(obs, next_obs, action_cvae, next_action_cvae, self.optimizer, loss_manager, valid_mode=valid_mode, device=self.device)
                        else:
                            loss = self.module.model.train_on_batch(obs, next_obs, self.optimizer, loss_manager, valid_mode=valid_mode, device=self.device)
                        # Loss: accumulate scalar loss
                        epoch_loss += loss
                        epoch_batches += 1
                        if not valid_mode:
                            # mean training loss so far
                            train_loss = epoch_loss / float(epoch_batches)
                        else:
                            # mean validation loss so far
                            val_loss = epoch_loss / float(epoch_batches)
                            val_loss_str = "{:.4f}*".format(
                                val_loss) if val_loss < best_error else "{:.4f}".format(val_loss)
                        if monitor_mode == 'loss':
                            if iter_ind % ITER_FLAG == 0 or (iter_ind == n_batch_per_epoch-1):
                                if not valid_mode:
                                    print("\rEpoch {:3}/{}, {:.2%}, train_loss: {:.4f} | (elapsed time: {:.2f}s)".format(
                                        epoch + 1, N_EPOCHS, (iter_ind+1)/n_batch_per_epoch, train_loss, time.time() - start_time), end="")
                                else:
                                    print("\r-------(valid): {:.2%}, val_loss: {} | (elapsed time: {:.2f}s)".format(
                                        (iter_ind+1)/n_batch_per_epoch, val_loss_str, time.time() - start_time), end="")
                        elif monitor_mode == 'pbar':
                            pbar.update(1)
                if valid_mode:
                    torch.set_grad_enabled(self.prev_grad_mode)
                if monitor_mode == 'loss':
                    print()
                elif monitor_mode == 'pbar':
                    pbar.close()
                    current_loss = val_loss if valid_mode else train_loss
                    print("Epoch {:3}/{}, {:.2%}, loss: {:.4f}".format(epoch + 1,
                                                                       N_EPOCHS, (iter_ind+1)/n_batch_per_epoch, current_loss))

            # Even if loss_history is modified by LossManager
            # we make it explicit

            def update_loss_history(loss_manager, train_loss, val_loss, epoch_batches, epoch):
                loss_history = loss_manager.loss_history
                loss_history['train_loss'].append(train_loss)
                loss_history['val_loss'].append(val_loss)
                for key in loss_history.keys():
                    if key in ['train_loss', 'val_loss']:
                        continue
                    loss_history[key][-1] /= epoch_batches
                    if epoch + 1 < N_EPOCHS:
                        loss_history[key].append(0)
                return loss_history
            loss_history = update_loss_history(loss_manager, train_loss, val_loss, epoch_batches, epoch)
            if self.model_type == 'gan' and not valid_mode:
                loss_history_D = update_loss_history(loss_manager_D, train_loss_D, 0, epoch_batches_D, epoch)
                loss_history_G = update_loss_history(loss_manager_G, train_loss_G, 0, epoch_batches_G, epoch)
            # Save best model
            if val_loss < best_error:
                best_error = val_loss
                torch.save(self.module.state_dict(), best_model_path)

            if np.isnan(train_loss):
                printRed("NaN Loss, consider increasing NOISE_STD in the gaussian noise layer")
                sys.exit(NAN_ERROR)
            # Then we print the results for this epoch:
            if (epoch + 1) % EPOCH_FLAG == 0:
                if figdir is not None:
                    self.module.eval()
                    with torch.no_grad():
                        # Optionally plot the current state space
                        print("Predicting states for all the observations...")
                        state_pred = self.predStatesWithDataLoader(dataloader_test)
                        plotRepresentation(state_pred, rewards,
                                           #    add_colorbar=epoch == 0,
                                           fit_pca=False,
                                           name="Learned State Representation (Training Data)",
                                           path=os.path.join(figdir_repr, "Epoch_{}.png".format(epoch+1)))
                        printGTC(state_pred, ground_truth, target_positions)
                        if self.use_autoencoder or self.use_vae or self.use_cvae or self.use_dae or self.model_type == "unet":  # or self.model_type == 'gan'
                            # Plot Reconstructed Image
                            if obs[0].shape[0] == 3:  # RGB
                                if self.use_cvae:
                                    reconstruct_obs = self.module.model.reconstruct(obs, action_cvae)
                                else:
                                    reconstruct_obs = self.module.model.reconstruct(obs)
                                # , normalize=True, range=(0,1)
                                print(obs[0].size())
                                print(obs[1].size())
                                print(reconstruct_obs[0].size())
                               
                                print(reconstruct_obs[1].size())
                               
                               
                                images = make_grid([obs[0], reconstruct_obs[0],obs[1], reconstruct_obs[1]], nrow=2)
                                plotImage(deNormalize(detachToNumpy(images)), mode='cv2',
                                          save2dir=figdir_recon, index=epoch+1)
                                # if self.use_dae:
                                #     raise NotImplementedError
                                #     plotImage(deNormalize(detachToNumpy(noisy_obs[0])), "Noisy Input Image (Train)")
                                # if self.perceptual_similarity_loss:
                                #     raise NotImplementedError
                                #     plotImage(deNormalize(detachToNumpy(decoded_obs_denoiser[0])),
                                #               "Reconstructed Image DAE")
                                #     plotImage(deNormalize(detachToNumpy(decoded_obs_denoiser_predicted[0])),
                                #               "Reconstructed Image predicted DAE")

                            elif obs[0].shape[0] % 3 == 0:  # Multi-RGB
                                raise NotImplementedError
                                for k in range(obs[0].shape[0] // 3):
                                    plotImage(deNormalize(detachToNumpy(obs[0][k * 3:(k + 1) * 3, :, :])),
                                              "Input Image {} (Train)".format(k + 1))
                                    # if self.use_dae:
                                    #     plotImage(deNormalize(detachToNumpy(noisy_obs[0][k * 3:(k + 1) * 3, :, :])),
                                    #               "Noisy Input Image (Train)".format(k + 1))
                                    # if self.perceptual_similarity_loss:
                                    #     plotImage(deNormalize(
                                    #         detachToNumpy(decoded_obs_denoiser[0][k * 3:(k + 1) * 3, :, :])),
                                    #         "Reconstructed Image DAE")
                                    #     plotImage(deNormalize(
                                    #         detachToNumpy(decoded_obs_denoiser_predicted[0][k * 3:(k + 1) * 3, :, :])),
                                    #         "Reconstructed Image predicted DAE")
                                    plotImage(deNormalize(detachToNumpy(reconstruct_obs[0][k * 3:(k + 1) * 3, :, :])),
                                              "Reconstructed Image {}".format(k + 1))

        # Load best model before predicting states
        self.module.load_state_dict(torch.load(best_model_path))

        print("Predicting states for all the observations...")
        # return predicted states for training observations
        self.module.eval()
        with torch.no_grad():
            pred_states = self.predStatesWithDataLoader(dataloader_test)
        pairs_loss_weight = [k for k in zip(loss_manager.names, loss_manager.weights)]
        if self.model_type == 'gan':
            pairs_loss_weight += [k for k in zip(loss_manager_D.names, loss_manager_D.weights)]
            pairs_loss_weight += [k for k in zip(loss_manager_G.names, loss_manager_G.weights)]
            # [Warning: the following line requires python >= 3.5]
            loss_history = {**loss_history, **loss_history_D, **loss_history_G}
        return loss_history, pred_states, pairs_loss_weight
