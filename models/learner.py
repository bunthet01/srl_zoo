
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
    kullbackLeiblerLossCCI, generationLoss, ganNonSaturateLoss, KLDloss, BCEloss
from losses.utils import findPriorsPairs
from pipeline import NAN_ERROR
from plotting.representation_plot import plotRepresentation, plotImage, printGTC
from preprocessing.data_loader import RobotEnvDataset
from preprocessing.utils import deNormalize, one_hot, attach_target_pos_to_all_imgs, sample_target_pos
from utils import printRed, detachToNumpy, printYellow
from .modules import SRLModules
from .priors import Discriminator as PriorDiscriminator
import sys
sys.path.append("..")
from real_robots.constants import TARGET_MAX_X, TARGET_MIN_X, TARGET_MAX_Y, TARGET_MIN_Y   # using Omnibot_env


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
    :param class_dim: (int) dimension of action
    :param use_conditional_model: (boolean) if cvae is used 
    :param only_action: (boolean) if we don't use target postion as codition to conditional model 
    :param use_cvae_new: (boolean) if cvae_new is used
    """

    def __init__(self, state_dim, class_dim, batch_size, seed=1, cuda=-1, use_conditional_model=False, only_action=False, use_cvae_new=False):
        super(BaseLearner, self).__init__()
        self.state_dim = state_dim
        self.class_dim = class_dim
        self.batch_size = batch_size
        self.module = None
        self.seed = seed
        self.use_dae = False
        self.use_conditional_model = use_conditional_model
        self.only_action = only_action
        self.use_cvae_new = use_cvae_new
        assert not self.use_dae, "Not implemented error."
        # Seed the random generator
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda >= 0:  # This will increase the training time by 5-10%.
            # Make CuDNN Determinist
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed(seed)

        self.device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() and (cuda >= 0) else "cpu")

    def _predFn(self, observations, action,target_pos,robot_pos, use_conditional_model, use_cvae_new):
        """
        Predict states in test mode given observations

        :param observations: (torch.Tensor)
        :return: (np.ndarray)
        """
        # Move the tensor back to the cpu
        if use_conditional_model:
            return detachToNumpy(self.module.model(observations, action, target_pos, self.only_action))
        elif use_cvae_new:
            return detachToNumpy(self.module.model(observations,target_pos,robot_pos))
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
            target_pos = batch[2].to(self.device)
            robot_pos = batch[3].to(self.device)
            predictions.append(self._predFn(obs, action, target_pos,robot_pos, self.use_conditional_model, self.use_cvae_new))

        return np.concatenate(predictions, axis=0)

    def predRewardsWithDataLoader(self, data_loader, split_dim_list=[2, 2], state_index=None, reward_name='reward'):
        """ 
        Predict rewards using minibatches to avoid memory issues
        :param data_loader: (DataLoader object)
        :return: (np.ndarray)
        """
        predictions = []
        gt_reward = []

        for obs, obs_next, rwd, action, next_action,target_pos, next_target_pos, robot_pos, next_robot_pos in data_loader:
            if reward_name == 'reward':
                obs = obs.to(self.device)
                if self.use_conditional_model:
                    action =  action.to(self.device)
                    target_pos  = target_pos.to(self.device)

                    states = self.module.model(obs, action, target_pos, self.only_action)
                elif self.use_cvae_new:
                    target_pos  = target_pos.to(self.device)
                    robot_pos = robot_pos.to(self.device)
                    states = self.module.model(obs, target_pos, robot_pos)
                else:
                    states = self.module.model(obs)
                if split_dim_list is not None:
                    states_split = torch.split(states, split_dim_list, dim=-1)
            obs_next = obs_next.to(self.device)
            if self.use_conditional_model:
                next_action = next_action.to(self.device)
                next_target_pos = next_target_pos.to(self.device)
                states_next = self.module.model(obs_next, next_action, next_target_pos, self.only_action)
            elif self.use_cvae_new:
                    next_target_pos  = next_target_pos.to(self.device)
                    next_robot_pos = next_robot_pos.to(self.device)
                    states_next = self.module.model(obs_next, next_target_pos, next_robot_pos)
            else:
                states_next = self.module.model(obs_next)
            if split_dim_list is not None:
                states_next_split = torch.split(states_next, split_dim_list, dim=-1)
            if reward_name == 'reward2':
                distance_state = (states_next_split[state_index[reward_name]] -
                                  states_next_split[state_index['inverse']])**2  # TODO TODO TODO
                rwd_pred = self.module.rewardModel2(distance_state)
            elif reward_name == 'reward':
                if state_index is not None:  # use split
                    rwd_pred = self.module.rewardModel(
                        states_split[state_index[reward_name]], states_next_split[state_index[reward_name]])
                else:
                    # no split
                    rwd_pred = self.module.rewardModel(states, states_next)
            else:
                raise NotImplementedError
            predictions.append(rwd_pred)
            gt_reward.append(rwd)
        return np.argmax(detachToNumpy(torch.cat(predictions, dim=0)), axis=-1), detachToNumpy(torch.cat(gt_reward, dim=0))

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
    :param beta: (float) for beta-vae, cci-vae, and cci-cvae
    :param use_cci: (bool) if cci is used
    :param Cmax: (float) for cci-vae or cci-cvae
    :param split_dimensions:
    :param path_to_dae: (str) path to pre-trained DAE when using perceptual loss
    :param state_dim_dae: (int)
    :param occlusion_percentage: (float) max percentage of occlusion when using DAE
    :param pretrained_weights_path: SRL pretrained model weights path (default: None)
    """

    def __init__(self, state_dim, class_dim, img_shape=None, model_type="resnet", inverse_model_type="linear", ls=False,add_noise=False,only_action=False,                                                  pretrained_weights_path=None,debug=False,log_folder="logs/default",
                 seed=1, learning_rate=0.001, learning_rate_gan=(0.001, 0.001), l1_reg=0.0, l2_reg=0.0, cuda=-1,
                 multi_view=False, losses=None, losses_weights_dict=None, n_actions=6, beta=1, Cmax=50, use_cci=False,
                 split_dimensions=-1, path_to_dae=None, state_dim_dae=200, occlusion_percentage=None):

        super(SRL4robotics, self).__init__(state_dim, class_dim, BATCH_SIZE, seed, cuda, use_conditional_model="cvae" in losses, only_action=only_action, use_cvae_new="cvae_new" in losses)

        self.multi_view = multi_view
        self.losses = losses
        self.dim_action = n_actions
        self.beta = beta
        self.use_cci = use_cci
        self.Cmax = Cmax
        self.ls = ls
        self.add_noise = add_noise
        self.only_action = only_action
        self.denoiser = None
        self.img_shape = img_shape
        self.model_type = model_type
        self.pretrained_weights_path = pretrained_weights_path
        self.cuda = cuda
        self.device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() and (cuda >= 0) else "cpu")
        if model_type in ["linear", "mlp", "resnet", "custom_cnn","custom_cnn_2", 'gan', 'unet','dc'] \
                or "autoencoder" in losses or "vae" in losses:
            self.use_forward_loss = "forward" in losses
            self.use_inverse_loss = "inverse" in losses
            self.use_reward_loss = "reward" in losses
            self.use_reward2_loss = "reward2" in losses
            self.no_priors = "priors" not in losses
            self.episode_prior = "episode-prior" in losses
            self.reward_prior = "reward-prior" in losses
            self.use_autoencoder = "autoencoder" in losses
            self.use_vae = "vae" in losses
            self.use_cvae = "cvae" in losses
            self.use_cvae_new = "cvae_new" in losses
            self.use_triplets = "triplet" in self.losses
            # self.perceptual_similarity_loss = "perceptual" in self.losses
            self.use_dae = "dae" in self.losses
            self.use_gan = "gan" in self.losses
            self.use_gan_new = "gan_new" in self.losses
            self.use_cgan = "cgan" in self.losses
            self.use_cgan_new = "cgan_new" in self.losses
            self.path_to_dae = path_to_dae

            if isinstance(split_dimensions, OrderedDict) and sum(split_dimensions.values()) > 0:
                printYellow("Using splitted representation")
                self.use_split = True
                self.split_dimensions = split_dimensions  # TODO UGLY!
            else:
                self.use_split = False
            self.module = SRLModules(state_dim=self.state_dim, class_dim = self.class_dim, img_shape=self.img_shape, action_dim=self.dim_action, model_type=model_type,
                                     losses=losses, split_dimensions=split_dimensions, inverse_model_type=inverse_model_type, device=self.device, only_action=self.only_action)
        else:
            raise ValueError("Unknown model: {}".format(model_type))

        print("Using {} model".format(model_type))

        

        if self.episode_prior:
            self.prior_discriminator = PriorDiscriminator(2 * self.state_dim).to(self.device)

        self.module = self.module.to(self.device)
        if self.use_gan or self.use_cgan or self.use_gan_new or self.use_cgan_new:
            assert not self.episode_prior, "NotImplementedError"
            self.optimizer_D = torch.optim.Adam(
                self.module.model.discriminator.parameters(), lr=learning_rate_gan[0], betas=(0.5, 0.9))
            self.optimizer_G = torch.optim.Adam(
                self.module.model.generator.parameters(), lr=learning_rate_gan[1], betas=(0.5, 0.9))
            if not self.use_gan_new and not self.use_cgan_new:
                combined_E_params = [param for param in self.module.model.encoder.parameters()]
                combined_E_params += [param for param in self.module.model.generator.parameters()]
                combined_E_params += [param for param in self.module.forward_net.parameters()]
                combined_E_params += [param for param in self.module.inverse_net.parameters()]
                combined_E_params += [param for param in self.module.reward_net.parameters()]
                combined_E_params += [param for param in self.module.classifier.parameters()]
                # import ipdb; ipdb.set_trace()
                self.optimizer = torch.optim.Adam(combined_E_params, lr=learning_rate, betas=(0.5, 0.9))
                # [TODO: check learnable parameters]
        else:
            learnable_params = [param for param in self.module.parameters() if param.requires_grad]
            # learnable_params = list(self.module.parameters()) ## NEW [TODO]

            if self.episode_prior:
                learnable_params += [p for p in self.prior_discriminator.parameters()]
            self.optimizer = torch.optim.Adam(learnable_params, lr=learning_rate)
        
        self.log_folder = log_folder

        # Default weights that are updated with the weights passed to the script
        self.losses_weights_dict = {"forward": 1.0, "inverse": 2.0, "reward": 100.0, "reward2": 100.0, "spcls": 100.0,
                                    "episode-prior": 1.0, "reward-prior": 10, "triplet": 1.0,
                                    "autoencoder": 1.0, "vae": 1.0, "cvae": 1.0, "perceptual": 1e-6, "dae": 1.0,
                                    'l1_reg': l1_reg, "l2_reg": l2_reg, 'random': 1.0}
        self.occlusion_percentage = occlusion_percentage
        self.state_dim_dae = state_dim_dae

        if losses_weights_dict is not None:
            self.losses_weights_dict.update(losses_weights_dict)

        self.debug = debug
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
              target_positions=None,
              truncate=None
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
            if self.use_gan_new or self.use_cgan_new:
                os.makedirs(figdir, exist_ok=True)
            else:
                figdir_repr = os.path.join(figdir, "state_repr")  # state representation scatter plot
                figdir_recon = os.path.join(figdir, "img_recon")  # image reconstruction folder
                os.makedirs(figdir_repr, exist_ok=True)
                os.makedirs(figdir_recon, exist_ok=True)

        sample_indices = np.arange(len(images_path))

        # Shuffle datasets
        sample_indices = sk_shuffle(sample_indices, random_state=0)
        valid_size = np.round(VALIDATION_SIZE * len(images_path)).astype(np.int64)
        indices_train, indices_val = sample_indices[:-valid_size], sample_indices[-valid_size:]
        
        all_imgs_target_pos = attach_target_pos_to_all_imgs(images_path, target_positions)     
        train_set = RobotEnvDataset(indices_train, images_path, actions,all_imgs_target_pos,relative_positions, rewards, episode_starts,
                                    mode=1, img_shape=self.img_shape, multi_view=self.multi_view,
                                    use_triplets=self.use_triplets, apply_occlusion=self.use_dae,
                                    occlusion_percentage=self.occlusion_percentage, dtype=np.float32)
        valid_set = RobotEnvDataset(indices_val, images_path, actions,all_imgs_target_pos,relative_positions, rewards, episode_starts,
                                    mode=1, img_shape=self.img_shape, multi_view=self.multi_view,
                                    use_triplets=self.use_triplets, apply_occlusion=self.use_dae,
                                    occlusion_percentage=self.occlusion_percentage, dtype=np.float32)
        test_set = RobotEnvDataset(np.arange(len(images_path)), images_path, actions,all_imgs_target_pos,relative_positions, rewards, episode_starts,
                                   mode=0, img_shape=self.img_shape, multi_view=self.multi_view,
                                   use_triplets=self.use_triplets, apply_occlusion=self.use_dae,
                                   occlusion_percentage=self.occlusion_percentage, dtype=np.float32)
        test_set2 = RobotEnvDataset(np.arange(len(images_path)), images_path, actions,all_imgs_target_pos,relative_positions, rewards, episode_starts,
                                    mode=2, img_shape=self.img_shape, multi_view=self.multi_view,
                                    use_triplets=self.use_triplets, apply_occlusion=self.use_dae,
                                    occlusion_percentage=self.occlusion_percentage, dtype=np.float32)

        data_loader_params = {'batch_size': self.batch_size,
                              'shuffle': True,
                              'num_workers': N_WORKERS,
                              #   'sampler': balanced_sampler,
                              'drop_last': True,
                              'pin_memory': False
                              }
        data_loader_params_test = {'batch_size': 128,
                                   'shuffle': False,
                                   'num_workers': N_WORKERS,
                                   #   'drop_last': False,
                                   'pin_memory': False
                                   }

        dataloader_train = torch.utils.data.DataLoader(train_set, **data_loader_params)
        dataloader_valid = torch.utils.data.DataLoader(valid_set, **data_loader_params)
        dataloader_test = torch.utils.data.DataLoader(test_set, **data_loader_params_test)
        dataloader_test2 = torch.utils.data.DataLoader(test_set2, **data_loader_params_test)
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

        # TRAINING -----------------------------------------------------------------------------------------------------
        loss_history = defaultdict(list)
        loss_manager = LossManager(self.module, loss_history)

        if self.use_gan or self.use_cgan or self.use_gan_new or self.use_cgan_new:
            loss_history_D = defaultdict(list)
            loss_history_G = defaultdict(list)
            
            loss_manager_D = LossManager(self.module.model.discriminator, loss_history_D)
            loss_manager_G = LossManager(self.module.model.generator, loss_history_G)
            # define the labels here so that we can reuse them (save time)
            label_valid = torch.ones((self.batch_size, 1)).to(self.device)
            label_fake = torch.zeros((self.batch_size, 1)).to(self.device)
        if self.use_gan_new or self.use_cgan_new:
            best_error_D = np.inf
            best_error_G = np.inf
            fixed_sample_state = torch.randn(self.batch_size, self.state_dim)
            fixed_sample_label = (torch.rand(self.batch_size, ) * self.class_dim).type(torch.LongTensor)
            fixed_sample_target_pos = sample_target_pos(self.batch_size,TARGET_MAX_X, TARGET_MIN_X, TARGET_MAX_Y, TARGET_MIN_Y)
        best_error = np.inf
        best_acc = -np.inf
        best_f1 = -np.inf
        best_model_path = "{}/srl_model.pth".format(self.log_folder)

        # Random features, we don't need to train a model
        if len(self.losses) == 1 and self.losses[0] == 'random':
            global N_EPOCHS
            N_EPOCHS = 0
            printYellow("Skipping training because using random features")
            torch.save(self.module.state_dict(), best_model_path)
        print("======================================================================================================================================")
        for epoch in range(N_EPOCHS):
            epoch_model_path = "{}/srl_model_{:04d}.pth".format(self.log_folder, epoch+1)
            if self.use_gan or self.use_cgan:
                epoch_batches_E_train = 0
                epoch_batches_D_train = 0
                epoch_batches_G_train = 0
            for valid_mode, dataloader in enumerate([dataloader_train, dataloader_valid]):
                if monitor_mode == 'pbar':
                    pbar = tqdm(total=len(dataloader))

                if self.use_gan or self.use_cgan or self.use_gan_new or self.use_cgan_new:
                    # GAN's training requires multi-optimizers, thus multiple loss_manager/epoch_loss/val_loss, etc.
                    epoch_loss_D, epoch_batches_D = 0, 0
                    epoch_loss_G, epoch_batches_G = 0, 0
                    epoch_loss_E, epoch_batches_E = 0, 0
                    d_acc, g_acc = 0, 0
                epoch_loss, epoch_batches = 0, 0       
                ep_rwd_acc, ep_inv_acc, ep_cls_acc = 0, 0, 0
                n_batch_per_epoch = len(dataloader)

                start_time = time.time()
                if valid_mode:
                    self.module.eval()
                    self.prev_grad_mode = torch.is_grad_enabled()
                    torch.set_grad_enabled(False)
                else:
                    self.module.train()
                dataloader = infinite_dataloader(dataloader)  # iter(dataloader)
                if self.debug:
                    n_batch_per_epoch = 1
                for iter_ind in range(n_batch_per_epoch):
                    (sample_idx, obs, next_obs, action, next_action, reward, cls_gt, target_pos, next_target_pos, robot_pos, next_robot_pos) = next(dataloader)
                    obs, next_obs = obs.to(self.device), next_obs.to(self.device)
                    cls_gt = cls_gt.to(self.device)

                    if not self.use_gan and not self.use_cgan: # gan need several update of encoder, so do not reset loss here !
                        # It's extremely important to release loss tensor, otherwise it will cause 'memory leak".
                        if not self.use_gan_new and not self.use_cgan_new:
                            self.optimizer.zero_grad()
                        loss_manager.resetLosses()

                    
                    # L1 regularization
                    if self.losses_weights_dict['l1_reg'] > 0:
                        l1Loss(loss_manager.reg_params,
                               self.losses_weights_dict['l1_reg'], loss_manager)
                    if self.losses_weights_dict['l2_reg'] > 0:
                        l2Loss(loss_manager.reg_params,
                               self.losses_weights_dict['l2_reg'], loss_manager)

                    # Actions associated to the observations
                    if self.use_forward_loss or self.use_inverse_loss or self.use_reward_loss or self.use_reward2_loss:
                        if self.use_cvae:
                            action = action.to(self.device)
                            next_action = next_action.to(self.device)
                            target_pos = target_pos.to(self.device)
                            next_target_pos = next_target_pos.to(self.device)
                            states, next_states = self.module.model(obs, action,target_pos, self.only_action), self.module.model(next_obs, next_action,next_target_pos,self.only_action)
                            actions_st = action.view(-1, 1).to(self.device)
                        elif self.use_cvae_new: 
                            action = action.to(self.device)
                            next_action = next_action.to(self.device)
                            robot_pos = robot_pos.to(self.device)
                            next_robot_pos = next_robot_pos.to(self.device)
                            target_pos = target_pos.to(self.device)
                            next_target_pos = next_target_pos.to(self.device)
                            states, next_states = self.module.model(obs,target_pos,robot_pos ), self.module.model(next_obs,next_target_pos,next_robot_pos)
                            actions_st = action.view(-1, 1).to(self.device)
                        else:
                            states, next_states = self.module(obs), self.module(next_obs)
                            actions_st = action.view(-1, 1).to(self.device)
                        

                    if not self.use_split:
                        if self.use_forward_loss:
                            self.module.add_forward_loss(states, actions_st, next_states,
                                                         loss_manager, weight=self.losses_weights_dict['forward'])
                        if self.use_inverse_loss:
                            self.module.add_inverse_loss(states, actions_st, next_states,
                                                         loss_manager, weight=self.losses_weights_dict['inverse'])
                        if self.use_reward_loss:
                            rewards_st = np.array(reward).copy()
                            # label are usually [-1, 0, 1] for non-shaped-reward
                            rewards_st = torch.from_numpy(rewards_st.astype(int)).to(self.device)
                            label_weights = torch.tensor([1.0, 100.0]).to(self.device)
                            # import ipdb; ipdb.set_trace()
                            self.module.add_reward_loss(states, rewards_st, next_states, loss_manager,
                                                        label_weights=label_weights, ignore_index=-1,  # TODO
                                                        weight=self.losses_weights_dict['reward'])
                        assert not self.use_reward2_loss, "reward2 model is only supported by 'split' srl model."
                        state_index = None
                        split_dim_list = None
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
                            name = "forward"
                            self.module.add_forward_loss(states_split_list[state_index["forward"]],
                                                         actions_st, next_states_split_list[state_index["forward"]
                                                                                            ], loss_manager,
                                                         weight=self.losses_weights_dict['forward'])
                        if self.use_inverse_loss:
                            name = "inverse"
                            self.module.add_inverse_loss(states_split_list[state_index["inverse"]],
                                                         actions_st, next_states_split_list[state_index["inverse"]
                                                                                            ], loss_manager,
                                                         weight=self.losses_weights_dict['inverse'])
                        if self.use_reward2_loss:
                            rewards_st = np.array(reward).copy()
                            rewards_st = 1-rewards_st
                            rewards_st = torch.from_numpy(rewards_st.astype(int)).to(self.device)
                            # label are usually [-1, 0, 1] for non-shaped-reward, now it's [2, 1, 0]
                            label_weights = torch.tensor([100.0, 1.0]).to(self.device)
                            # distance_state should be estimated for 'next_state'
                            distance_state = (next_states_split_list[state_index["reward2"]
                                                                     ] - next_states_split_list[state_index["inverse"]])**2
                            self.module.add_reward2_loss(distance_state,
                                                         rewards_st,
                                                         loss_manager,
                                                         label_weights=label_weights,
                                                         ignore_index=2,
                                                         weight=self.losses_weights_dict['reward2'])
                            self.module.add_spcls_loss(states_split_list[state_index["reward2"]],
                                                       cls_gt,
                                                       loss_manager,
                                                       weight=self.losses_weights_dict['spcls'])
                        if self.use_reward_loss:
                            rewards_st = np.array(reward).copy()
                            # label are usually [-1, 0, 1] for non-shaped-reward
                            rewards_st = torch.from_numpy(rewards_st.astype(int)).to(self.device)
                            label_weights = torch.tensor([1.0, 100.0]).to(self.device)
                            self.module.add_reward_loss(states_split_list[state_index["reward"]],
                                                        rewards_st,
                                                        next_states_split_list[state_index["reward"]],
                                                        loss_manager,
                                                        label_weights=label_weights, ignore_index=-1,
                                                        weight=self.losses_weights_dict['reward'])

                    if self.use_gan or self.use_cgan or self.use_gan_new or self.use_cgan_new:
                        if self.use_gan_new or self.use_cgan_new:
                            if self.use_gan_new:
                                loss_D, loss_G, history_message = self.module.model.train_on_batch(obs,epoch_batches,figdir,epoch,fixed_sample_state, self.optimizer_D, 
                                                                                               self.optimizer_G, loss_manager_D, 
                                                                                               loss_manager_G , valid_mode=valid_mode, 
                                                                                               device=self.device, label_smoothing=self.ls,add_noise=self.add_noise, minibatch=n_batch_per_epoch)
                            else:
                                loss_D, loss_G, history_message = self.module.model.train_on_batch(obs,action,target_pos,epoch_batches,figdir,epoch,fixed_sample_state,fixed_sample_label,
                                                                                               fixed_sample_target_pos, self.optimizer_D, 
                                                                                               self.optimizer_G, loss_manager_D, 
                                                                                               loss_manager_G , valid_mode=valid_mode, 
                                                                                               device=self.device, label_smoothing=self.ls,add_noise=self.add_noise,only_action=self.only_action, minibatch=n_batch_per_epoch)
                                
                            epoch_loss_D +=loss_D
                            epoch_loss_G +=loss_G
                            epoch_batches += 1
                            if not valid_mode:
                                train_loss = 0
                                train_loss_D = epoch_loss_D/ float(epoch_batches)
                                train_loss_G = epoch_loss_G/ float(epoch_batches)
                            else:
                                val_loss = 0
                                val_loss_D = epoch_loss_D/ float(epoch_batches)
                                val_loss_G = epoch_loss_G/ float(epoch_batches)

                        else:
                            loss, history_message, epoch_loss_E, epoch_batches_E,epoch_loss_D,epoch_batches_D,epoch_loss_G,epoch_batches_G = self.module.model.train_on_batch(
                                                                                    obs, next_obs,
                                                                                    self.optimizer_D, self.optimizer_G, self.optimizer,
                                                                                    loss_manager_D, loss_manager_G, loss_manager,
                                                                                    epoch_batches_D, epoch_batches_G, epoch_batches_E,
                                                                                    epoch_loss_D, epoch_loss_G, epoch_loss_E,
                                                                                    d_acc, g_acc,
                                                                                    batch_size=self.batch_size, 
                                                                                    dataloader=dataloader,
                                                                                    valid_mode=valid_mode,
                                                                                    device=self.device,only_action = self.only_action)
                            epoch_loss += loss
                            epoch_batches += 1 ## update batch count
                            if not valid_mode:
                                # mean training loss so far. For gan, it's already the mean.
                                train_loss = loss
                                train_loss_D = epoch_loss_D / float(epoch_batches_D)
                                train_loss_G = epoch_loss_G / float(epoch_batches_G)
                                
                                epoch_batches_E_train = epoch_batches_E
                                epoch_batches_D_train = epoch_batches_D
                                epoch_batches_G_train = epoch_batches_G
                                
                                
                                
                            else:
                                # mean validation loss so far. For gan, it's already the mean.
                                val_loss = loss
                            # Custom/Optional plots: training too long, display/save img during training.
                            if iter_ind % 20 == 0 and not valid_mode:
                                if self.use_cgan:
                                    reconstruct_obs = self.module.model.reconstruct(obs, action, target_pos, self.only_action)
                                else:
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
                    # elif: ## ===========  add special srl model here. ===========
                    else:  # ============= Main update for usual srl model ====================
                        # Compute weighted average of losses of encoder part (including 'forward'/'inverse'/'reward' models)
                        if self.use_cci:
                            if not valid_mode:
                                c = self.Cmax*(iter_ind+epoch*len(dataloader_train))/(N_EPOCHS*len(dataloader_train))
                        else: c = 0
                        
                        if self.use_cvae:
                            action = action.to(self.device)
                            next_action = next_action.to(self.device)
                            target_pos = target_pos.to(self.device)
                            next_target_pos = next_target_pos.to(self.device)
                            loss = self.module.model.train_on_batch(obs, next_obs, action, next_action,target_pos, next_target_pos, self.optimizer, loss_manager, valid_mode, self.device, self.beta, c,self.only_action )
                        elif self.use_cvae_new:
                            target_pos = target_pos.to(self.device)
                            next_target_pos = next_target_pos.to(self.device)
                            robot_pos = robot_pos.to(self.device)
                            next_robot_pos = next_robot_pos.to(self.device)
                            
                            loss = self.module.model.train_on_batch(obs, next_obs,target_pos, next_target_pos,robot_pos, next_robot_pos, self.optimizer, loss_manager, valid_mode, self.device, self.beta, c)
                        elif self.use_vae:
                            loss = self.module.model.train_on_batch(obs, next_obs, self.optimizer, loss_manager, valid_mode, self.device, self.beta, c)
                        else:
                            loss = self.module.model.train_on_batch(obs, next_obs, self.optimizer, loss_manager, valid_mode=valid_mode, device=self.device)
                        history_message = ""
                        epoch_loss += loss
                        epoch_batches += 1
                        if not valid_mode:
                            # mean training loss so far
                            train_loss = epoch_loss / float(epoch_batches)
                        else:
                            # mean validation loss so far
                            val_loss = epoch_loss / float(epoch_batches)
                    # Accuracy
                    if self.use_split:
                        with torch.no_grad():
                            if self.use_inverse_loss:
                                # HACK it's not a good way using 'if' like this.
                                name = "inverse"
                                state_pred = states_split_list[state_index[name]]
                                next_state_pred = next_states_split_list[state_index[name]]
                                act_pred = self.module.inverseModel(state_pred, next_state_pred)
                                act_pred = torch.argmax(act_pred, dim=-1)
                                inv_acc = torch.sum(actions_st.view(-1) == act_pred).float() / actions_st.numel()
                                inv_acc = inv_acc.item()
                            elif self.use_forward_loss:  # only forward loss without inverse loss
                                # HACK it's not a good way using 'if' like this.
                                name = "forward"
                                state_pred = states_split_list[state_index[name]]
                                inv_acc = 0
                            else:
                                inv_acc = 0
                            if self.use_reward2_loss:
                                state_pred = states_split_list[state_index["reward2"]]
                                distance_state = (
                                    next_states_split_list[state_index["reward2"]] - next_state_pred)**2
                                rwd_pred = self.module.rewardModel2(distance_state.detach())
                                rwd_pred = torch.argmax(rwd_pred, dim=-1)
                                rwd_acc = torch.sum(rewards_st == rwd_pred).float() / rewards_st.numel()
                                rwd_acc = rwd_acc.item()

                                cls_pred = self.module.classifier(state_pred.detach())
                                cls_pred = torch.argmax(cls_pred, dim=-1)
                                cls_acc = torch.sum(cls_pred == cls_gt).float() / cls_gt.numel()
                                cls_acc = cls_acc.item()
                            if self.use_reward_loss:
                                state_pred = states_split_list[state_index["reward"]]
                                next_state_pred = next_states_split_list[state_index["reward"]]
                                rwd_pred = self.module.rewardModel(state_pred.detach(), next_state_pred.detach())
                                rwd_pred = torch.argmax(rwd_pred, dim=-1)
                                rwd_acc = torch.sum(rewards_st == rwd_pred).float() / rewards_st.numel()
                                rwd_acc = rwd_acc.item()
                                # raise NotImplementedError
                                cls_acc = 0.0
                            if (not self.use_reward2_loss) and (not self.use_reward_loss):
                                rwd_acc = 0.0
                                cls_acc = 0.0
                    else:
                        assert not self.use_reward2_loss, "reward2 is only supported by split model."
                        cls_acc = 0.0
                        if self.use_inverse_loss:
                            act_pred = self.module.inverseModel(states, next_states)
                            act_pred = torch.argmax(act_pred, dim=-1)
                            inv_acc = torch.sum(actions_st.view(-1) == act_pred).float() / actions_st.numel()
                            inv_acc = inv_acc.item()
                        else:
                            inv_acc = 0.0
                        if self.use_reward_loss:
                            rwd_pred = self.module.rewardModel(states.detach(), next_states.detach())
                            rwd_pred = torch.argmax(rwd_pred, dim=-1)
                            rwd_acc = torch.sum(rewards_st == rwd_pred).float() / rewards_st.numel()
                            rwd_acc = rwd_acc.item()
                            # raise NotImplementedError
                        else:
                            rwd_acc = 0.0

                    # Loss: accumulate scalar loss
                    ep_rwd_acc += rwd_acc
                    ep_inv_acc += inv_acc
                    ep_cls_acc += cls_acc
                    
                    if valid_mode:
                        if self.use_gan_new or self.use_cgan_new:
                            val_loss_D_str =  "{:.4f}**".format(val_loss_D) if val_loss_D < best_error_D else "{:.4f}".format(val_loss_D)
                            val_loss_G_str =  "{:.4f}**".format(val_loss_G) if val_loss_G < best_error_G else "{:.4f}".format(val_loss_G)
                        else:
                            val_rwd_acc = ep_rwd_acc / float(epoch_batches)
                            val_inv_acc = ep_inv_acc / float(epoch_batches)
                            val_cls_acc = ep_cls_acc / float(epoch_batches)
                            val_acc = (val_rwd_acc + val_inv_acc + val_cls_acc) / 3.
                            val_loss_str = "{:.4f}**".format(
                                val_loss) if val_loss < best_error else "{:.4f}".format(val_loss)
                            val_acc_str = "{:.4f}**".format(
                                  val_acc) if val_acc > best_acc else "{:.4f}".format(val_acc)
                        
                    if monitor_mode == 'loss':
                        if iter_ind % ITER_FLAG == 0 or (iter_ind == n_batch_per_epoch-1):
                            if self.use_gan_new or self.use_cgan_new:
                                if not valid_mode:
                                    print("\rEpoch {:3}/{}, {:.2%}, train_loss_D: {:.4f} train_loss_G: {:.4f}|supp: {}| (time: {:.2f}s)".format(
                                        epoch + 1, N_EPOCHS, (iter_ind+1)/n_batch_per_epoch, train_loss_D, train_loss_G , history_message, time.time() - start_time), end="")
                                else:
                                    print("\r-------(valid): {:.2%}, val_loss_D: {} val_loss_G: {}|supp: {}| (time: {:.2f}s)".format(
                                        (iter_ind+1)/n_batch_per_epoch, '___', '___', history_message, time.time() - start_time), end="")
                            else:
                                if not valid_mode:
                                    print("\rEpoch {:3}/{}, {:.2%}, train_loss: {:.4f} rwd_acc: {:.2%} inv_acc: {:.2%} cls_acc: {:.2%}|supp: {}| (time: {:.2f}s)".format(
                                        epoch + 1, N_EPOCHS, (iter_ind+1)/n_batch_per_epoch, train_loss, rwd_acc, inv_acc, cls_acc, history_message, time.time() - start_time), end="")
                                else:
                                    print("\r-------(valid): {:.2%}, val_loss: {} val_acc: {}|supp: {}| (time: {:.2f}s)".format(
                                        (iter_ind+1)/n_batch_per_epoch, val_loss_str, val_acc_str, history_message, time.time() - start_time), end="")
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
            # we make it explicit ??
            
            def update_loss_history(loss_manager, train_loss,name_train_loss, val_loss, name_val_loss, epoch_batches, epoch):
                loss_history = loss_manager.loss_history
                loss_history[name_train_loss].append(train_loss)
                loss_history[name_val_loss].append(val_loss)
                for key in loss_history.keys():
                    if key in [name_train_loss, name_val_loss]:
                        continue
                    loss_history[key][-1] /= float(epoch_batches)
                    if epoch + 1 < N_EPOCHS:
                        loss_history[key].append(0)
                return loss_history

            if self.use_gan or self.use_cgan:      
                loss_history = update_loss_history(loss_manager, train_loss,"train_loss_E", val_loss, "val_loss_E", epoch_batches_E_train, epoch)
                loss_history_D = update_loss_history(loss_manager_D, train_loss_D,"train_loss_D", 0, "val_loss_D", epoch_batches_D_train, epoch)
                loss_history_G = update_loss_history(loss_manager_G, train_loss_G,"train_loss_G", 0, "val_loss_G", epoch_batches_G_train, epoch)
            elif self.use_gan_new or self.use_cgan_new:
                loss_history_D = update_loss_history(loss_manager_D, train_loss_D,"train_loss_D", val_loss_D, "val_loss_D", len(dataloader_train), epoch)
                loss_history_G = update_loss_history(loss_manager_G, train_loss_G,"train_loss_G", val_loss_G, "val_loss_G", len(dataloader_train), epoch)
            else:
                loss_history = update_loss_history(loss_manager, train_loss,"train_loss", val_loss, "val_loss", len(dataloader_train), epoch)
            # Save best model
            if self.use_gan_new or self.use_cgan_new:
                torch.save(self.module.state_dict(), epoch_model_path)
                if val_loss_D < best_error_D:  # TODO TODO TODO
                    best_error_D = val_loss_D
                    if not self.use_split:
                        torch.save(self.module.state_dict(), best_model_path)  # save weight at each epoch !
                if val_loss_G < best_error_G:  # TODO TODO TODO
                    best_error_G = val_loss_G
                    if not self.use_split:
                        torch.save(self.module.state_dict(), best_model_path)  # save weight at each epoch !
            else:   
                if val_loss < best_error:  # TODO TODO TODO
                    best_error = val_loss
                    if not self.use_split:
                        torch.save(self.module.state_dict(), best_model_path)  # save weight at each epoch !
                if val_acc > best_acc:  # TODO TODO TODO
                    best_acc = val_acc
                    if not self.use_split:
                        torch.save(self.module.state_dict(), best_model_path)  # save weight at each epoch !
                    # torch.save(self.module.state_dict(), best_model_path)
                if self.use_split:
                    # save weight at each epoch ! # TODO TODO TODO TODO
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
                            if self.use_reward2_loss and self.use_split:  # reward2 only compatible with split
                                reward_pred, reward_gt = self.predRewardsWithDataLoader(dataloader_test2,
                                                                                        state_index=state_index,
                                                                                        split_dim_list=split_dim_list,
                                                                                        reward_name='reward2')
                                reward_gt = 1-reward_gt
                                f1_a = np.sum(((reward_gt-reward_pred) == 0) * (reward_pred == 0))
                                f1_b = np.sum(((reward_pred-reward_gt) == -1) * (reward_pred == 0))
                                f1_c = np.sum(((reward_pred-reward_gt) != 0) * (reward_gt == 0))
                                f1_score = 2*f1_a/(2*f1_a+f1_b+f1_c)
                                recall = f1_a / (f1_a + f1_c)
                            elif self.use_reward_loss:
                                reward_pred, reward_gt = self.predRewardsWithDataLoader(dataloader_test2,
                                                                                        state_index=state_index,
                                                                                        split_dim_list=split_dim_list,
                                                                                        reward_name='reward')
                                f1_a = np.sum(((reward_gt-reward_pred) == 0) * (reward_pred == 1))
                                f1_b = np.sum(((reward_pred-reward_gt) == 1) * (reward_pred == 1))
                                f1_c = np.sum(((reward_pred-reward_gt) != 0) * (reward_gt == 1))
                                f1_score = 2*f1_a/(2*f1_a+f1_b+f1_c)
                                recall = f1_a / (f1_a + f1_c)

                            plotRepresentation(state_pred, rewards,
                                               #    add_colorbar=epoch == 0,
                                               fit_pca=False,
                                               name="Learned State Representation (Training Data)",
                                               path=os.path.join(figdir_repr, "Epoch_{}.png".format(epoch+1)))
                            printGTC(state_pred, ground_truth, target_positions, truncate=truncate)
                            if self.use_autoencoder or self.use_vae or self.use_cvae or self.use_cvae_new or self.use_dae or self.use_gan or self.use_cgan:
                                # Plot Reconstructed Image
                                if obs[0].shape[0] == 3:  # RGB
                                    if self.use_cvae or self.use_cgan:
                                        reconstruct_obs = self.module.model.reconstruct(obs, action.to(self.device), target_pos.to(self.device),self.only_action)
                                    elif self.use_cvae_new:
                                        reconstruct_obs = self.module.model.reconstruct(obs,target_pos.to(self.device),robot_pos.to(self.device))
                                    else:
                                        reconstruct_obs = self.module.model.reconstruct(obs)
                                    # , normalize=True, range=(0,1)
                                    images = make_grid([obs[0], reconstruct_obs[0], obs[1], reconstruct_obs[1]], nrow=2)
                                    plotImage(deNormalize(detachToNumpy(images)), mode='cv2',
                                              save2dir=figdir_recon, index=epoch+1)
                        if (self.use_reward2_loss and self.use_split) or self.use_reward_loss:
                            if f1_score > best_f1:
                                best_f1 = f1_score
                                # torch.save(self.module.state_dict(), best_model_path) # TODO TODO TODO TODO
                                f1_score_str = "{:.4f}***".format(f1_score)
                            else:
                                f1_score_str = "{:.4f}".format(f1_score)
                            print("F1 score: {}; Recall: {:.4f}".format(f1_score_str, recall))
            print("======================================================================================================================================")
        
        # Load best model before predicting states
        self.module.load_state_dict(torch.load(best_model_path))
        if not self.use_gan_new and not self.use_cgan_new:
            print("Predicting states for all the observations...")
            # return predicted states for training observations
            self.module.eval()
            with torch.no_grad():
                pred_states = self.predStatesWithDataLoader(dataloader_test)
            pairs_loss_weight = [k for k in zip(loss_manager.names, loss_manager.weights)]
            if self.use_gan or self.use_cgan :
                pairs_loss_weight += [k for k in zip(loss_manager_D.names, loss_manager_D.weights)]
                pairs_loss_weight += [k for k in zip(loss_manager_G.names, loss_manager_G.weights)]
                loss_history = [loss_history, loss_history_D, loss_history_G]
        else: 
            pairs_loss_weight = [k for k in zip(loss_manager_D.names, loss_manager_D.weights)]
            pairs_loss_weight += [k for k in zip(loss_manager_G.names, loss_manager_G.weights)]
            loss_history = [loss_history_D, loss_history_G]
            pred_states = None
        return loss_history, pred_states, pairs_loss_weight
