# coding: utf-8
"""
This is a PyTorch implementation of the method for state representation learning described in the paper "Learning State
Representations with Robotic Priors" (Jonschkowski & Brock, 2015).

This program is based on the original implementation by Rico Jonschkowski (rico.jonschkowski@tu-berlin.de):
https://github.com/tu-rbo/learning-state-representations-with-robotic-priors

Example to run this program:
 python main.py --path slot_car_task_train.npz

# Some details:
-Weight initialization: Xavier method (by )default for Conv layers https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L40)
TODO: generator to load images on the fly
"""
# preventing incompatibility errors:
#  https://docs.python.org/3/howto/pyporting.html#prevent-compatibility-regressions
from __future__ import print_function, division

import argparse
import time
import json

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

from plotting.representation_plot import plot_representation, plt

# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

try:
    from functools import reduce
except ImportError:
    pass

DISPLAY_PLOTS = True
EPOCH_FLAG = 1  # Plot every 1 epoch
BATCH_SIZE = 256  #
NOISE_STD = 1e-6  # To avoid NaN (states must be different)
MAX_BACTHSIZE_GPU = 512  # For plotting, max batch_size before having memory issues


def observationsGenerator(observations, batch_size=64, cuda=False):
    """
    Python generator to avoid out of memory issues
    when predicting states for all the observations
    :param observations: (torch tensor)
    :param batch_size: (int)
    :param cuda: (bool)
    """
    n_minibatches = len(observations) // batch_size + 1
    for i in range(n_minibatches):
        start_idx, end_idx = batch_size * i, batch_size * (i + 1)
        obs_var = Variable(observations[start_idx:end_idx], volatile=True)
        if cuda:
            obs_var = obs_var.cuda()
        yield obs_var


class SRLConvolutionalNetwork(nn.Module):
    """
    Convolutional Neural Net for State Representation Learning (SRL)
    input shape : 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
    :param state_dim: (int)
    :param batch_size: (int)
    :param cuda: (bool)
    """

    def __init__(self, state_dim=2, batch_size=256, cuda=False):
        super(SRLConvolutionalNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.squeezeNet = models.squeezenet1_0(pretrained=True)
        # Freeze params
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Replace the last fully-connected layer
        n_units = self.resnet.fc.in_features
        print("{} units in the last layer".format(n_units))
        self.resnet.fc = nn.Linear(n_units, state_dim)
        if cuda:
            self.resnet.cuda()
        self.noise = GaussianNoise(batch_size, state_dim, NOISE_STD, cuda=cuda)

    def forward(self, x):
        x = self.resnet(x)
        x = self.noise(x)
        return x


class SRLDenseNetwork(nn.Module):
    """
    Feedforward Neural Net for State Representation Learning (SRL)
    input shape : 3-channel RGB images of shape (3 x H x W) (to be consistent with CNN network)
    :param input_dim: (int) 3 x H x H
    :param state_dim: (int)
    :param batch_size: (int)
    :param cuda: (bool)
    :param n_hidden: (int)
    """

    def __init__(self, input_dim, state_dim=2,
                 batch_size=256, cuda=False, n_hidden=32):
        super(SRLDenseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_hidden)
        self.fc2 = nn.Linear(n_hidden, state_dim)
        self.noise = GaussianNoise(batch_size, state_dim, NOISE_STD, cuda=cuda)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.noise(x)
        return x


class GaussianNoise(nn.Module):
    """
    Gaussian Noise layer
    :param batch_size: (int)
    :param input_dim: (int)
    :param std: (float) standard deviation
    :param mean: (float)
    :param cuda: (bool)
    Noise is not part of data augmentation, it just prevents NaN when using robotic priors
    """

    def __init__(self, batch_size, input_dim, std, mean=0, cuda=False):
        super(GaussianNoise, self).__init__()
        self.std = std
        self.mean = mean
        self.noise = Variable(th.zeros(batch_size, input_dim))
        if cuda:
            self.noise = self.noise.cuda()

    def forward(self, x):
        if self.training:
            self.noise.data.normal_(self.mean, std=self.std)
            return x + self.noise
        return x


class RoboticPriorsLoss(nn.Module):
    """
    :param model: (PyTorch model)
    :param l1_reg: (float) l1 regularization coeff
    """
    def __init__(self, model, l1_reg=0.0):
        super(RoboticPriorsLoss, self).__init__()
        # Retrieve only trainable and regularizable parameters (we should exclude biases)
        self.reg_params = [param for name, param in model.named_parameters() if
                           ".bias" not in name and param.requires_grad]
        n_params = sum([reduce(lambda x, y: x * y, param.size()) for param in self.reg_params])
        self.l1_coeff = (l1_reg / n_params)

    def forward(self, states, next_states, states_from_same_pos_as_states, dissimilar_pairs, same_actions_pairs):
        """
        :param states: (th Variable)
        :param next_states: (th Variable)
        :param dissimilar_pairs: (th tensor)
        :param same_actions_pairs: (th tensor)
        :param same_states_diff_norm: (th tensor)
        :return: (th Variable)
        """
        state_diff = next_states - states
        state_diff_norm = state_diff.norm(2, dim=1)
        similarity = lambda x, y: th.exp(-(x - y).norm(2, dim=1) ** 2)
        temp_coherence_loss = (state_diff_norm ** 2).mean()
        causality_loss = similarity(states[dissimilar_pairs[:, 0]],
                                    states[dissimilar_pairs[:, 1]]).mean()
        proportionality_loss = ((state_diff_norm[same_actions_pairs[:, 0]] -
                                 state_diff_norm[same_actions_pairs[:, 1]]) ** 2).mean()

        repeatability_loss = (
            similarity(states[same_actions_pairs[:, 0]], states[same_actions_pairs[:, 1]]) *
            (state_diff[same_actions_pairs[:, 0]] - state_diff[same_actions_pairs[:, 1]]).norm(2, dim=1) ** 2).mean()

        # 5th prior assumes all sequences in the dataset share at least one same 3D pos input image of Baxter arm
        same_pos_states_diff = states - states_from_same_pos_as_states
        same_pos_states_diff_norm = same_pos_states_diff.norm(2, dim=1)
        fixed_ref_point_loss = (same_pos_states_diff_norm ** 2).mean()

        l1_loss = sum([th.sum(th.abs(param)) for param in self.reg_params])

        loss = 1 * temp_coherence_loss + 1 * causality_loss + 5 * proportionality_loss \
               + 5 * repeatability_loss + 1 * fixed_ref_point_loss + self.l1_coeff * l1_loss
        return loss


class SRL4robotics:
    """
    :param state_dim: (int)
    :param model_type: (str) one of "cnn" or "mlp"
    :param seed: (int)
    :param learning_rate: (float)
    :param l1_reg: (float)
    :param cuda: (bool)
    """

    def __init__(self, state_dim, model_type="cnn", log_folder="logs/default",
                 seed=1, learning_rate=0.001, l1_reg=0.0, cuda=False):

        self.state_dim = state_dim
        self.batch_size = BATCH_SIZE
        self.cuda = cuda

        np.random.seed(seed)
        th.manual_seed(seed)
        if cuda:
            th.cuda.manual_seed(seed)

        if model_type == "cnn":
            self.model = SRLConvolutionalNetwork(self.state_dim, self.batch_size, cuda)
        elif model_type == "mlp":
            input_dim = 224 * 224 * 3
            self.model = SRLDenseNetwork(input_dim, self.state_dim, self.batch_size, cuda)
        else:
            raise ValueError("Unknown model: {}".format(model_type))
        print("Using {} model".format(model_type))

        if cuda:
            self.model.cuda()
        learnable_params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = th.optim.Adam(learnable_params, lr=learning_rate)
        self.l1_reg = l1_reg
        self.log_folder = log_folder

    def _predFn(self, observations, restore_train=True):
        """
        Predict states in test mode given observations
        :param observations: (PyTorch Variable)
        :param restore_train: (bool) whether to restore training mode after prediction
        :return: (numpy tensor)
        """
        # Switch to test mode
        self.model.eval()
        states = self.model(observations)
        if restore_train:
            # Restore training mode
            self.model.train()
        if self.cuda:
            # Move the tensor back to the cpu
            return states.data.cpu().numpy()
        return states.data.numpy()

    def predStates(self, observations):
        """
        Predict states for given observations
        WARNING: you should use _batchPredStates
        if observations tensor is large to avoid memory issues
        :param observations: (numpy tensor)
        :return: (numpy tensor)
        """
        observations = observations.astype(np.float32)
        obs_var = Variable(th.from_numpy(observations), volatile=True)
        if self.cuda:
            obs_var = obs_var.cuda()
        states = self._predFn(obs_var, restore_train=False)
        return states

    def _batchPredStates(self, observations):
        """
        Predict states using minibatches to avoid memory issues
        :param observations: (numpy tensor)
        :return: (numpy tensor)
        """
        predictions = []
        for obs_var in observationsGenerator(th.from_numpy(observations), MAX_BACTHSIZE_GPU, cuda=self.cuda):
            predictions.append(self._predFn(obs_var))
        return np.concatenate(predictions, axis=0)

    def learn(self, observations, actions, rewards, episode_starts, data_folder):
        """
        Learn a state representation
        :param observations: (numpy tensor)
        :param actions: (np matrix) of each action id performed. To support
            Jonschkowski's race_car baseline, each action id is within an array: e.g. [[4], [8], [2]]
        :param rewards: (numpy 1D array)
        :param episode_starts: (numpy 1D array) boolean array
                                the ith index is True if one episode starts at this frame
        :return: (numpy tensor) the learned states for the given observations
        """

        # PREPARE DATA -------------------------------------------------------------------------------------------------
        # here, we organize the data into minibatches
        # and find pairs for the respective loss terms

        # We assume that observations are already preprocessed
        # that is to say normalized and scaled
        observations = observations.astype(np.float32)

        num_samples = observations.shape[0] - 1  # number of samples

        # indices for all time steps where the episode continues
        indices = np.array([i for i in range(num_samples) if not episode_starts[i + 1]], dtype='int64')
        np.random.shuffle(indices)

        # split indices into minibatches. minibatchlist is a list of lists; each
        # list is the id of the observation preserved thorough the training
        minibatchlist = [np.array(sorted(indices[start_idx:start_idx + self.batch_size]))
                         for start_idx in range(0, num_samples - self.batch_size + 1, self.batch_size)]
        if len(minibatchlist[-1]) < self.batch_size:
            print("Removing last minibatch of size {} < batch_size".format(len(minibatchlist[-1])))
            del minibatchlist[-1]

        find_same_actions = lambda index, minibatch: \
            np.where(np.prod(actions[minibatch] == actions[minibatch[index]], axis=1))[0]

        same_actions = [   #TODO: DOCUMENT THE DATA TYPE HERE, IT IS NOT TRIVIAL, e.g.: list of arrays, each containing one pair of observation ids
            np.array([[i, j] for i in range(self.batch_size) for j in find_same_actions(i, minibatch) if j > i],
                     dtype='int64') for minibatch in minibatchlist]

        # Here we save which (observation index) samples should be dissimilar because they lead
        # to different rewards after the same actions. The * represents the mask
        # of 0s and 1s indicating the fulfilment of the two conditions (= AND gate)
        # Final [0] gives the array of row indexes where condition is true ([1] gives columns)
        # np.prod transforms the Jonschkowski required format of action ids
        # actions = [[1],[6],[4]] -> {1, 6, 4]}
        find_dissimilar = lambda index, minibatch: \
            np.where(np.prod(actions[minibatch] == actions[minibatch[index]], axis=1) *
                     (rewards[minibatch + 1] != rewards[minibatch[index] + 1]))[0]

        dissimilar = [np.array([[i, j] for i in range(self.batch_size) for j in find_dissimilar(i, minibatch) if j > i],
                               dtype='int64') for minibatch in minibatchlist]
        #print('actions {}, same actions {} and np.prod {}', actions[minibatch], actions[minibatch[index]], actions[minibatch] == actions[minibatch[index]])


        if fixed_ref_point in arm_states:
            print('Exact fixed ref point found in arm_states, no need to add threshold for finding equivalent grount truth arm_states (arm_position)')
        find_same_ref_point_position = lambda index, minibatch: \
            np.where(np.prod(arm_states[minibatch] == arm_states[minibatch[index]], axis=1))[0]
            # 0 returns the row indexes ([1] for column indexes) of the cases where the (prod) resulting matrix satisfied the where condition

        same_ref_points = [
            np.array([[i, j] for i in range(self.batch_size) for j in find_same_ref_points(i, minibatch) if j > i],
                     dtype='int64') for minibatch in minibatchlist]

        for item in same_actions + dissimilar:
            if len(item) == 0:
                msg = "No similar or dissimilar pairs found for at least one minibatch (currently is {})\n".format(BATCH_SIZE)
                msg += "=> Consider increasing the batch_size or changing the seed"
                raise ValueError(msg)
        for item in same_ref_points:
            if len(item) == 0:
                msg = "No same ref point position observation of the arm was found for at least one minibatch (currently is {})\n".format(BATCH_SIZE)
                msg += "=> Consider increasing the batch_size or changing the seed\n same_ref_point_positions: {}, arm_states:{}, fixed_ref_point:{}".format(same_ref_points, arm_states, fixed_ref_point)
                raise ValueError(msg)


        # TRAINING -----------------------------------------------------------------------------------------------------
        criterion = RoboticPriorsLoss(self.model, self.l1_reg)
        best_error = np.inf

        self.model.train()
        start_time = time.time()
        n_batches = len(minibatchlist)
        for epoch in range(N_EPOCHS):
            # In each epoch, we do a full pass over the training data:
            epoch_loss, epoch_batches = 0, 0
            enumerated_minibatches = list(enumerate(minibatchlist))  # print('minibatchlist and enum_mini {}   {}'.format(minibatchlist, enumerated_minibatches))
            # shuffle the order of the minibatchlist while preserving the indexes for each minibatch
            np.random.shuffle(enumerated_minibatches)
            for i, batch in enumerated_minibatches: #                print('i, batch from enum_mini {}  {}  {}'.format(i, batch, enumerated_minibatches))
                diss = dissimilar[i][np.random.permutation(dissimilar[i].shape[0])]
                same = same_actions[i][
                    np.random.permutation(same_actions[i].shape[0])]  # [:MAX_PAIR_PER_SAMPLE * self.batch_size]
                same_point = same_ref_points[i][
                    np.random.permutation(same_ref_points[i].shape[0])]
                diss, same, same_point = th.from_numpy(diss), th.from_numpy(same), th.from_numpy(same_point)
                obs = Variable(th.from_numpy(observations[batch]))
                next_obs = Variable(th.from_numpy(observations[batch + 1]))
                # select a random but different batch index (non necessarily consecutive either)
                batch_random_index = np.random.permutation(np.delete(np.arange(i), i))[0] # Equiv to something like np.random.choice(n_batches- the index of current batch i, 1, replace=False)
                obs_from_same_pos_as_obs = Variable(th.from_numpy(observations[same_point[batch_random_index]]))

                if self.cuda:
                    obs, next_obs, obs_from_same_pos_as_obs = obs.cuda(), next_obs.cuda(), obs_from_same_pos_as_obs.cuda()
                    same, diss = same.cuda(), diss.cuda()

                states, next_states, states_from_same_pos_as_states = self.model(obs), self.model(next_obs), self.model(obs_from_same_pos_as_obs)
                self.optimizer.zero_grad()
                loss = criterion(states, next_states, states_from_same_pos_as_states, diss, same)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data[0]
                epoch_batches += 1

            # Save best model
            # TODO: use a validation set
            if epoch_loss / epoch_batches < best_error:
                th.save(self.model.state_dict(), "{}/srl_model.pyth.pkl".format(self.log_folder))

            # Then we print the results for this epoch:
            if (epoch + 1) % EPOCH_FLAG == 0:
                print("Epoch {:3}/{}, loss:{:.4f}".format(epoch + 1, N_EPOCHS, epoch_loss / epoch_batches))
                print("{:.2f}s/epoch".format((time.time() - start_time) / (epoch + 1)))
                if DISPLAY_PLOTS:
                    # Optionally plot the current state space
                    plot_representation(self._batchPredStates(observations), rewards, add_colorbar=epoch == 0,
                                        name="Learned State Representation (Training Data)")
        if DISPLAY_PLOTS:
            plt.close("Learned State Representation (Training Data)")

        # return predicted states for training observations
        return self._batchPredStates(observations)

def saveStates(states, images_path, rewards, log_folder):
    """
    Save learned states to json and npz files
    :param states: (numpy array)
    :param images_path: ([str])
    :param rewards: (rewards)
    :param log_folder: (str)
    """
    print("Saving image path to state representation")
    image_to_state = {path: list(map(str, state)) for path, state in zip(images_path, states)}
    with open("{}/image_to_state.json".format(log_folder), 'wb') as f:
        json.dump(image_to_state, f, sort_keys=True)
    print("Saving states and rewards")
    states_rewards = {'states': states, 'rewards': rewards}
    np.savez('{}/states_rewards.npz'.format(log_folder), **states_rewards)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch SRL with robotic priors')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--state_dim', type=int, default=2, help='state dimension (default: 2)')
    parser.add_argument('-bs', '--batch_size', type=int, default=256, help='batch_size (default: 256)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.005, help='learning rate (default: 0.005)')
    parser.add_argument('--l1_reg', type=float, default=0.0, help='L1 regularization coeff (default: 0.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-plots', action='store_true', default=False, help='disables plots')
    parser.add_argument('--model_type', type=str, default="cnn", help='Model architecture (default: "cnn")')
    parser.add_argument('--path', type=str, default="", help='Path to npz file', required=True)
    parser.add_argument('--data_folder', type=str, default="", help='Dataset folder')
    parser.add_argument('--log_folder', type=str, default='logs/default_folder',
                        help='Folder within logs/ where the experiment model and plots will be saved')
    parser.add_argument('--ref_prior', action='store_true', default=False,
                        help='Applies 5th Fixed Reference Point Prior')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and th.cuda.is_available()
    DISPLAY_PLOTS = not args.no_plots
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    print('\nDataset npz file: {}\n'.format(args.path))
    print('Log folder: {}'.format(args.log_folder))

    print('Loading data ... ')
    training_data = np.load(args.path)
    observations, actions = training_data['observations'], training_data['actions']
    rewards, episode_starts = training_data['rewards'], training_data['episode_starts']

    if args.data_folder == "":
        raise ValueError("Fifth prior cannot be applied if --data_folder parameter for ground truth states are not provided")
    # Demo with Rico's original data
    if len(observations.shape) == 2:
        import cv2
        from preprocessing.preprocess import IMAGE_WIDTH, IMAGE_HEIGHT
        from preprocessing.utils import preprocessInput

        observations = observations.reshape(-1, 16, 16, 3) * 255.
        obs = np.zeros((observations.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, 3))
        for i in range(len(observations)):
            obs[i] = cv2.resize(observations[i], (IMAGE_WIDTH, IMAGE_HEIGHT))
        del observations
        observations = preprocessInput(obs, mode="image_net")

    # Move the channel dimension to match pretrained model input
    # (batch_size, width, height, n_channels) -> (batch_size, n_channels, height, width)
    observations = np.transpose(observations, (0, 3, 2, 1))
    print("Observations shape: {}".format(observations.shape))

    print('Learning a state representation ... ')
    srl = SRL4robotics(args.state_dim, model_type=args.model_type, seed=args.seed,
                       log_folder=args.log_folder, learning_rate=args.learning_rate,
                       l1_reg=args.l1_reg, cuda=args.cuda)
    if srl.ref_prior and 'slot_car_task_train' in args.data_folder:
        raise ValueError("Jonschkowski's racing slot_car_car baseline is not supported \
                         to apply the 5th prior on as it does not contain ground truth")
    learned_states = srl.learn(observations, actions, rewards, episode_starts, args.data_folder)

    # We should always save the states learned, why only if dataset is given?
    if args.data_folder != "":
        ground_truth = np.load("data/{}/ground_truth.npz".format(args.data_folder))
        saveStates(learned_states, ground_truth['images_path'], rewards, args.log_folder)

    name = "Learned State Representation\n {}".format(args.log_folder.split('/')[-1])
    path = "{}/learned_states.png".format(args.log_folder)
    plot_representation(learned_states, rewards, name, add_colorbar=True, path=path)

    if DISPLAY_PLOTS: # needed to keep the plots showing while training
        input('\nPress any key to exit.')
