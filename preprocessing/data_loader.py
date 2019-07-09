from __future__ import print_function, division, absolute_import

import glob
import random
import time

# Python 2/3 support
try:
    import queue
except ImportError:
    import Queue as queue

import cv2
import numpy as np
import torch as th
from joblib import Parallel, delayed
from torch.multiprocessing import Queue, Process


from .utils import preprocessInput
import torch.utils.data


def sample_coordinates(coord_1, max_distance, percentage):
    """
    Sampling from a coordinate A, a second one B within a maximum distance [max_distance X percentage]

    :param coord_1: (int) sample first coordinate
    :param max_distance: (int) max value of coordinate in the axis
    :param percentage: (float) maximum occlusion as a percentage
    :return: (tuple of int)
    """
    min_coord_2 = max(0, coord_1 - max_distance * percentage)
    max_coord_2 = min(coord_1 + max_distance * percentage, max_distance)
    coord_2 = np.random.randint(low=min_coord_2, high=max_coord_2)
    return min(coord_1, coord_2), max(coord_1, coord_2)


def preprocessImage(image, img_reshape=None, convert_to_rgb=True, apply_occlusion=False, occlusion_percentage=0.5):
    """
    :param image: (np.ndarray) image (BGR or RGB)
    :param img_reshape: (None or tuple e.g. (3, 128, 128)) reshape image to (128, 128)
    :param convert_to_rgb: (bool) whether the conversion to rgb is needed or not
    :param apply_occlusion: (bool) whether to occludes part of the images or not
                            (used for training denoising autoencoder)
    :param occlusion_percentage: (float) max percentage of occlusion (in width and height)
    :return: (np.ndarray)
    """
    # Resize
    if img_reshape is not None:
        assert isinstance(img_reshape, tuple), "'img_reshape' should be a tuple like: (3,128,128)"
        assert img_reshape[0] < 10, "'img_reshape' should be a tuple like: (3,128,128)"
        im = cv2.resize(image, img_reshape[1:], interpolation=cv2.INTER_AREA)
    else:
        im = image
        img_reshape = (im.shape[-1],) + im.shape[:-1]
    # Convert BGR to RGB
    if convert_to_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # Normalize
    im = preprocessInput(im.astype(np.float32))

    img_height, img_width = img_reshape[1:]
    if apply_occlusion:
        h_1 = np.random.randint(img_height)
        h_1, h_2 = sample_coordinates(h_1, img_height, percentage=occlusion_percentage)
        w_1 = np.random.randint(img_width)
        w_1, w_2 = sample_coordinates(w_1, img_width, percentage=occlusion_percentage)
        noisy_img = im
        # This mask is set by applying zero values to corresponding pixels.
        noisy_img[h_1:h_2, w_1:w_2, :] = 0.
        im = noisy_img

    return im


class DataLoader(object):
    def __init__(self, minibatchlist, images_path, img_shape=None, n_workers=1, multi_view=False, use_triplets=False,
                 infinite_loop=True, max_queue_len=4, is_training=False, apply_occlusion=False,
                 occlusion_percentage=0.5):
        """
        A Custom dataloader to work with our datasets, and to prepare data for the different models
        (inverse, priors, autoencoder, ...)

        :param minibatchlist: ([np.array]) list of observations indices (grouped per minibatch)
        :param images_path: (np.array) Array of path to images
        :param n_workers: (int) number of preprocessing worker (load and preprocess each image)
        :param multi_view: (bool)
        :param use_triplets: (bool)
        :param infinite_loop: (bool) whether to have an iterator that can be resetted, set to False, it
        :param max_queue_len: (int) Max number of minibatches that can be preprocessed at the same time
        :param apply_occlusion: is the use of occlusion enabled - when using DAE (bool)
        :param occlusion_percentage: max percentage of occlusion when using DAE (float)
        :param is_training: (bool)
        :param img_shape: (tuple or None) if None, image will not be resize, else: resize image to new shape (channels first)) e.g. img_shape = (3, 128, 128).

            Set to True, the dataloader will output both `obs` and `next_obs` (a tuple of th.Tensor)
            Set to false, it will only output one th.Tensor.
        """
        super(DataLoader, self).__init__()
        self.n_workers = n_workers
        self.infinite_loop = infinite_loop
        self.n_minibatches = len(minibatchlist)
        self.minibatchlist = minibatchlist
        self.images_path = images_path
        self.shuffle = is_training
        self.queue = Queue(max_queue_len)
        self.process = None
        self.use_triplets = use_triplets
        self.multi_view = multi_view
        # apply occlusion for training a DAE
        self.apply_occlusion = apply_occlusion
        self.occlusion_percentage = occlusion_percentage
        self.img_shape = img_shape
        self.startProcess()

    @staticmethod
    def createTestMinibatchList(n_samples, batch_size):
        """
        Create list of minibatch for plotting
        :param n_samples: (int)
        :param batch_size: (int)
        :return: ([np.array])
        """
        minibatchlist = []
        for i in range(n_samples // batch_size + 1):
            start_idx = i * batch_size
            end_idx = min(n_samples, (i + 1) * batch_size)
            minibatchlist.append(np.arange(start_idx, end_idx))
        return minibatchlist

    def startProcess(self):
        """Start preprocessing process"""
        self.process = Process(target=self._run)
        # Make it a deamon, so it will be deleted at the same time
        # of the main process
        self.process.daemon = True
        self.process.start()

    def _run(self):
        start = True
        with Parallel(n_jobs=self.n_workers, batch_size="auto", backend="threading") as parallel:
            while start or self.infinite_loop:
                start = False

                if self.shuffle:
                    indices = np.random.permutation(self.n_minibatches).astype(np.int64)
                else:
                    indices = np.arange(len(self.minibatchlist), dtype=np.int64)

                for minibatch_idx in indices:
                    batch_noisy, batch_obs_noisy, batch_next_obs_noisy = None, None, None
                    if self.shuffle:
                        images = np.stack((self.images_path[self.minibatchlist[minibatch_idx]],
                                           self.images_path[self.minibatchlist[minibatch_idx] + 1]))
                        images = images.flatten()
                    else:
                        images = self.images_path[self.minibatchlist[minibatch_idx]]

                    if self.n_workers <= 1:
                        batch = [self._makeBatchElement(image_path, self.multi_view, self.use_triplets, img_shape=self.img_shape)
                                 for image_path in images]
                        if self.apply_occlusion:
                            batch_noisy = [self._makeBatchElement(image_path, self.multi_view, self.use_triplets,
                                                                  apply_occlusion=self.apply_occlusion,
                                                                  occlusion_percentage=self.occlusion_percentage,
                                                                  img_shape=self.img_shape)
                                           for image_path in images]

                    else:
                        batch = parallel(delayed(self._makeBatchElement)(image_path, self.multi_view,
                                                                         self.use_triplets, img_shape=self.img_shape) for image_path in images)
                        if self.apply_occlusion:
                            batch_noisy = parallel(
                                delayed(self._makeBatchElement)(image_path, self.multi_view, self.use_triplets,
                                                                apply_occlusion=self.apply_occlusion,
                                                                occlusion_percentage=self.occlusion_percentage,
                                                                img_shape=self.img_shape)
                                for image_path in images)

                    batch = th.cat(batch, dim=0)
                    if self.apply_occlusion:
                        batch_noisy = th.cat(batch_noisy, dim=0)

                    if self.shuffle:
                        batch_obs, batch_next_obs = batch[:len(images) // 2], batch[len(images) // 2:]
                        if batch_noisy is not None:
                            batch_obs_noisy, batch_next_obs_noisy = batch_noisy[:len(images) // 2], \
                                batch_noisy[len(images) // 2:]
                        self.queue.put((minibatch_idx, batch_obs, batch_next_obs,
                                        batch_obs_noisy, batch_next_obs_noisy))
                    else:
                        self.queue.put(batch)

                    # Free memory
                    if self.shuffle:
                        del batch_obs
                        del batch_next_obs
                        if batch_noisy is not None:
                            del batch_obs_noisy
                            del batch_next_obs_noisy
                    del batch
                    del batch_noisy

                self.queue.put(None)

    @classmethod
    def _makeBatchElement(cls, image_path, multi_view=False, use_triplets=False, apply_occlusion=False,
                          occlusion_percentage=None, img_shape=None):
        """
        :param image_path: (str) path to an image (without the 'data/' prefix)
        :param multi_view: (bool)
        :param use_triplets: (bool)
        :return: (th.Tensor)
        """
        # Remove trailing .jpg if present
        image_path = 'srl_zoo/data/' + image_path.split('.jpg')[0]

        if multi_view:
            images = []

            # Load different view of the same timestep
            for i in range(2):
                im = cv2.imread("{}_{}.jpg".format(image_path, i + 1))
                if im is None:
                    raise ValueError("tried to load {}_{}.jpg, but it was not found".format(image_path, i + 1))
                images.append(preprocessImage(im, img_reshape=img_shape, apply_occlusion=apply_occlusion,
                                              occlusion_percentage=occlusion_percentage))
            ####################
            # loading a negative observation
            if use_triplets:
                # End of file format for positive & negative observations (camera 1) - length : 6 characters
                extra_chars = '_1.jpg'

                # getting path for all files of same record episode, e.g path_to_data/record_001/frame[0-9]{6}*
                digits_path = glob.glob(image_path[:-6] + '[0-9]*' + extra_chars)

                # getting the current & all frames' timesteps
                current = int(image_path[-6:])
                # For all others extract last 6 digits (timestep) after removing the extra chars
                all_frame_steps = [int(k[:-len(extra_chars)][-6:]) for k in digits_path]
                # removing current positive timestep from the list
                all_frame_steps.remove(current)

                # negative timestep by random sampling
                length_set_steps = len(all_frame_steps)
                negative = all_frame_steps[random.randint(0, length_set_steps - 1)]
                negative_path = '{}{:06d}'.format(image_path[:-6], negative)

                im3 = cv2.imread(negative_path + "_1.jpg")
                if im3 is None:
                    raise ValueError("tried to load {}_{}.jpg, but it was not found".format(negative_path, 1))
                im3 = preprocessImage(im3, img_reshape=img_shape)
                # stacking along channels
                images.append(im3)

            im = np.dstack(images)
        else:
            im = cv2.imread("{}.jpg".format(image_path))
            if im is None:
                raise ValueError("tried to load {}.jpg, but it was not found".format(image_path))

            im = preprocessImage(im, img_reshape=img_shape, apply_occlusion=apply_occlusion,
                                 occlusion_percentage=occlusion_percentage)

        # Channel first (for pytorch convolutions) + one dim for the batch
        # th.tensor creates a copy
        im = th.tensor(im.reshape((1,) + im.shape).transpose(0, 3, 1, 2))
        return im

    def __len__(self):
        return self.n_minibatches

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                val = self.queue.get_nowait()
                break
            except queue.Empty:
                time.sleep(0.001)
                continue
        if val is None:
            raise StopIteration
        return val

    next = __next__  # Python 2 compatibility

    def __del__(self):
        if self.process is not None:
            self.process.terminate()


class SupervisedDataLoader(DataLoader):
    """
    Data loader for supervised learning.
    :param x_indices: (np.array) indices of observations
    :param y_values: (np.array) targets for each input value
    :param images_path: (np.array) Array of path to images
    :param batch_size: (int)
    :param n_workers: (int) number of workers used for preprocessing
    :param no_targets: (bool) Set to true, only inputs are generated
    :param shuffle: (bool) Set to True, the dataloader will shuffle the indices
    :param infinite_loop: (bool) whether to have an iterator that can be resetted, set to False, it
    :param max_queue_len: (int) Max number of minibatches that can be preprocessed at the same time
    """

    def __init__(self, x_indices, y_values, images_path, batch_size, img_shape=None, n_workers=1, no_targets=False,
                 shuffle=False, infinite_loop=True, max_queue_len=4):
        # Create minibatch list
        minibatchlist, targets = self.createMinibatchList(x_indices, y_values, batch_size)

        # Whether to yield targets together with output
        # (not needed when plotting or predicting states)
        self.no_targets = no_targets
        self.targets = np.array(targets)
        self.shuffle = shuffle
        super(SupervisedDataLoader, self).__init__(minibatchlist, images_path, img_shape=img_shape, n_workers=n_workers,
                                                   infinite_loop=infinite_loop, max_queue_len=max_queue_len)

    def _run(self):
        start = True
        with Parallel(n_jobs=self.n_workers, batch_size="auto", backend="threading") as parallel:
            while start or self.infinite_loop:
                start = False
                if self.shuffle:
                    indices = np.random.permutation(self.n_minibatches).astype(np.int64)
                else:
                    indices = np.arange(len(self.minibatchlist), dtype=np.int64)

                for minibatch_idx in indices:
                    images = self.images_path[self.minibatchlist[minibatch_idx]]

                    if self.n_workers <= 1:
                        batch = [self._makeBatchElement(image_path, img_shape=self.img_shape) for image_path in images]
                    else:
                        batch = parallel(delayed(self._makeBatchElement)(
                            image_path, img_shape=self.img_shape) for image_path in images)

                    batch = th.cat(batch, dim=0)

                    if self.no_targets:
                        self.queue.put(batch)
                    else:
                        # th.tensor creates a copy
                        self.queue.put((batch, th.tensor(self.targets[minibatch_idx])))

                    # Free memory
                    del batch

                self.queue.put(None)

    @staticmethod
    def createMinibatchList(x_indices, y_values, batch_size):
        """
        Create list of minibatches (contains the observations indices)
        along with the corresponding list of targets
        Warning: this may create minibatches of different lengths

        :param x_indices: (np.array)
        :param y_values: (np.array)
        :param batch_size: (int)
        :return: ([np.array], [np.array])
        """
        targets = []
        minibatchlist = []
        n_minibatches = len(x_indices) // batch_size + 1
        for i in range(0, n_minibatches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(x_indices))
            excerpt = slice(start_idx, end_idx)
            # Remove excerpt with no elements
            if len(x_indices[excerpt]) > 0:
                minibatchlist.append(x_indices[excerpt])
                targets.append(y_values[excerpt])

        return minibatchlist, targets


class RobotEnvDataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""
    """
    A Custom dataloader to work with our datasets, and to prepare data for the different models
    (inverse, priors, autoencoder, ...)

    :param minibatchlist: ([np.array]) list of observations indices (grouped per minibatch)
    :param images_path: (np.array) Array of path to images
    :param n_workers: (int) number of preprocessing worker (load and preprocess each image)
    :param multi_view: (bool)
    :param use_triplets: (bool)
    :param infinite_loop: (bool) whether to have an iterator that can be resetted, set to False, it
    :param max_queue_len: (int) Max number of minibatches that can be preprocessed at the same time
    :param apply_occlusion: is the use of occlusion enabled - when using DAE (bool)
    :param occlusion_percentage: max percentage of occlusion when using DAE (float)
    :param is_training: (bool)
    :param img_shape: (tuple or None) if None, image will not be resize, else: resize image to new shape (channels first)) e.g. img_shape = (3, 128, 128).

        Set to True, the dataloader will output both `obs` and `next_obs` (a tuple of th.Tensor)
        Set to false, it will only output one th.Tensor.
    """

    def __init__(self, sample_indices, images_path, actions, rewards, episode_starts,
                 img_shape=None,
                 is_training=True,
                 multi_view=False,
                 use_triplets=False,
                 apply_occlusion=False,
                 occlusion_percentage=0.5,
                 dtype=np.float32,
                 img_extension="jpg"):
        super(RobotEnvDataset, self).__init__()
        # Initialization
        self.sample_indices = sample_indices
        self.images_path = images_path
        self.actions = actions
        self.rewards = rewards
        self.episode_starts = episode_starts

        self.img_shape = img_shape
        self.is_training = is_training
        self.use_triplets = use_triplets
        self.multi_view = multi_view
        # apply occlusion for training a DAE
        self.apply_occlusion = apply_occlusion
        self.occlusion_percentage = occlusion_percentage

        self.dtype = dtype
        self.img_extension = img_extension
    def __len__(self):
        ## 'Denotes the total number of samples'
        return len(self.sample_indices)

    def _get_one_img(self, image_path):
        # self.minibatchlist = minibatchlist
        image_path = 'data/' + image_path.split('.{}'.format(self.img_extension))[0]  # [TODO]

        img = cv2.imread("{}.{}".format(image_path, self.img_extension))
        if img is None:
            raise ValueError("tried to load {}.{}, but it was not found".format(image_path, self.img_extension))
        img = preprocessImage(img, img_reshape=self.img_shape,
                              apply_occlusion=self.apply_occlusion,
                              occlusion_percentage=self.occlusion_percentage)
        img = img.transpose(2, 0, 1)
        return img

    def __getitem__(self, index):
        # 'Generates one sample of data': (main)

        index = self.sample_indices[index]  # real index of samples
        if (index+1) >= len(self.actions) or self.episode_starts[index + 1]:
            # the case where 'index' is the end of episode, no next observation.
            index -= 1  # this may repeat some observations, but the proba is rare.

        image_path = self.images_path[index]
        # Load data and get label
        if not self.multi_view:
            if self.is_training:
                img = self._get_one_img(image_path)
                img_next = self._get_one_img(self.images_path[index+1])
                action = self.actions[index]
                next_action = self.actions[index+1]
                reward = self.rewards[index]
                return index, img.astype(self.dtype), img_next.astype(self.dtype), action, next_action, reward, 1, 1
            else:
                img = self._get_one_img(image_path)
                action = self.actions[index]
                return img.astype(self.dtype), action

        else:  # [TODO: not tested yet]
            raise NotImplementedError
            images = []
            # Load different view of the same timestep
            for i in range(2):
                img = cv2.imread("{}_{}.jpg".format(image_path, i + 1))
                if img is None:
                    raise ValueError("tried to load {}_{}.jpg, but it was not found".format(image_path, i + 1))
                images.append(preprocessImage(img, img_reshape=self.img_shape, apply_occlusion=self.apply_occlusion,
                                              occlusion_percentage=self.occlusion_percentage))
            ####################
            # loading a negative observation
            if use_triplets:
                # End of file format for positive & negative observations (camera 1) - length : 6 characters
                extra_chars = '_1.jpg'
                # getting path for all files of same record episode, e.g path_to_data/record_001/frame[0-9]{6}*
                digits_path = glob.glob(image_path[:-6] + '[0-9]*' + extra_chars)
                # getting the current & all frames' timesteps
                current = int(image_path[-6:])
                # For all others extract last 6 digits (timestep) after removing the extra chars
                all_frame_steps = [int(k[:-len(extra_chars)][-6:]) for k in digits_path]
                # removing current positive timestep from the list
                all_frame_steps.remove(current)

                # negative timestep by random sampling
                length_set_steps = len(all_frame_steps)
                negative = all_frame_steps[random.randint(0, length_set_steps - 1)]
                negative_path = '{}{:06d}'.format(image_path[:-6], negative)

                im3 = cv2.imread(negative_path + "_1.jpg")
                if im3 is None:
                    raise ValueError("tried to load {}_{}.jpg, but it was not found".format(negative_path, 1))
                im3 = preprocessImage(im3, img_reshape=img_shape)
                # stacking along channels
                images.append(im3)
            img = np.dstack(images)

            return img.astype(self.dtype)  # , y.astype(self.dtype)#.to(self.dtype), y.to(self.dtype)
        return img.astype(self.dtype)

    @staticmethod
    def preprocessLabels(labels):
        # (Min, Max) labels[:, 0] = (0.452, 3.564)
        # (Min, Max) labels[:, 1] = (0.222, 3.795)
        # set to Min = 0.15, Max = 3.85
        val_max = 3.85  # np.max(labels, axis=0)
        val_min = 0.15  # np.min(labels, axis=0)
        labels = (labels - val_min) / (val_max - val_min)
        labels = 2*labels - 1

        return labels

    @staticmethod
    def createTestMinibatchList(n_samples, batch_size):
        """
        Create list of minibatch for plotting
        :param n_samples: (int)
        :param batch_size: (int)
        :return: ([np.array])
        """
        minibatchlist = []
        for i in range(n_samples // batch_size + 1):
            start_idx = i * batch_size
            end_idx = min(n_samples, (i + 1) * batch_size)
            minibatchlist.append(np.arange(start_idx, end_idx))
        return minibatchlist

def convertScalerToVectorAction(x):
    """
    :param x: np.ndarray
    :return:
    """
    counter = 0
    for i in x:
        if i == 0:
            l = np.array([1, 0, 0, 0])
        elif i == 1:
            l = np.array([0, 1, 0, 0])
        elif i == 2:
            l = np.array([0, 0, 1, 0])
        else :
            l = np.array([0, 0, 0, 1])
        if counter == 0:
            result = [l]
        else:
            result = np.append(result, [l], 0)
        counter = counter+1
    return result