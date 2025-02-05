
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
from torch.utils.data import Sampler
try:
    from sklearn.model_selection import StratifiedShuffleSplit
except:
    print('Need scikit-learn for this functionality')


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


class BalancedLabelSampler(Sampler):
    r"""Balanced classes sampler (batch-level)
    Arguments:
        data_source (Dataset): dataset to sample from
        class_labels (np.ndarray): 
    """
    def __init__(self, data_source, class_labels, subset=None, batch_size=32):
        self.data_source = data_source
        self.class_labels = class_labels
        self.batch_size = batch_size
        bin_count = np.bincount(self.class_labels)
        self.num_classes = len(bin_count)
        ## Statistic abount dataset
        print("Find {} classes: max/min samples per class: {}/{}; variation amplitude (std/max) {:.2%}".format(\
            self.num_classes, np.max(bin_count), np.min(bin_count), np.std(bin_count)/np.max(bin_count)))
        assert len(class_labels) == len(self.data_source), "class_labels should have same length as dataset."
        # class_dict = 
        num_batches = int(len(self.data_source)/self.batch_size)
        resampling = []
        replacement = not (self.num_classes > self.batch_size)
        for ind in range(num_batches):
            sample_classes = np.random.choice(np.arange(self.num_classes), self.batch_size, replace=replacement)
            for cls_label in sample_classes:
                return
            # resampling

    def __iter__(self):
        return 
    def __len__(self):
        return len(self.data_source)


class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size, subset=None):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector
        self.subset = subset

    def gen_sample_array(self):

        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = th.randn(self.class_vector.size(0), 2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)

class DataLoader(object):
    def __init__(self, minibatchlist, images_path,img_shape=None, n_workers=1, multi_view=False, use_triplets=False,
                 infinite_loop=True, max_queue_len=4, is_training=False, apply_occlusion=False,
                 occlusion_percentage=0.5, absolute_path=False):
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
        Set to True, the dataloader will output both `obs` and `next_obs` (a tuple of th.Tensor)
        Set to false, it will only output one th.Tensor.
        """
        super(DataLoader, self).__init__()
        self.n_workers = n_workers
        self.infinite_loop = infinite_loop
        self.n_minibatches = len(minibatchlist)
        self.minibatchlist = minibatchlist
        self.images_path = images_path
        self.img_shape = img_shape
        self.shuffle = is_training
        self.queue = Queue(max_queue_len)
        self.process = None
        self.use_triplets = use_triplets
        self.multi_view = multi_view
        # apply occlusion for training a DAE
        self.apply_occlusion = apply_occlusion
        self.occlusion_percentage = occlusion_percentage
        self.absolute_path = absolute_path
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
                        batch = [self._makeBatchElement(image_path, self.img_shape, self.multi_view, self.use_triplets,
                                                                  absolute_path=self.absolute_path)
                                 for image_path in images]
                        if self.apply_occlusion:
                            batch_noisy = [self._makeBatchElement(image_path,self.img_shape, self.multi_view, self.use_triplets,
                                                                  apply_occlusion=self.apply_occlusion,
                                                                  occlusion_percentage=self.occlusion_percentage,
                                                                  absolute_path=self.absolute_path)
                                           for image_path in images]

                    else:
                        batch = parallel(
                            delayed(self._makeBatchElement)(image_path, self.img_shape, self.multi_view, self.use_triplets,
                                                                  absolute_path=self.absolute_path)
                            for image_path in images)
                        if self.apply_occlusion:
                            batch_noisy = parallel(delayed(self._makeBatchElement)(image_path, self.img_shape, self.multi_view, self.use_triplets,
                                                                apply_occlusion=self.apply_occlusion,
                                                                occlusion_percentage=self.occlusion_percentage,
                                                                absolute_path=self.absolute_path) for image_path in images)

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
    def _makeBatchElement(cls, image_path, img_shape, multi_view=False, use_triplets=False, apply_occlusion=False,
                          occlusion_percentage=None, absolute_path=False):
        """
        :param image_path: (str) path to an image (without the 'data/' prefix)
        :param multi_view: (bool)
        :param use_triplets: (bool)
        :return: (th.Tensor)
        """
        # Remove trailing .jpg if present
        prepath = '' if absolute_path else 'data/'
        image_path = prepath + image_path.split('.jpg')[0]

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

            im = preprocessImage(im, img_reshape=img_shape, apply_occlusion=apply_occlusion, occlusion_percentage=occlusion_percentage)

        # Channel first (for pytorch convolutions) + one dim for the batch
        # th.tensor creates a copy
        im = th.tensor(im.reshape((1,) + im.shape).transpose(0, 3, 2, 1))
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


class DataLoaderConditional(object):
    def __init__(self, minibatchlist,actions,task, generative_model_state_dim,TARGET_MAX_X, TARGET_MIN_X, TARGET_MAX_Y, TARGET_MIN_Y, MAX_X, MIN_X, MAX_Y, MIN_Y,grid_walker, grid_list,target_pos_list, seed, max_queue_len=4, infinite_loop=True):
        """
        A Custom dataloader preparing data to forward to Conditional model   
        :param n_workers: (int) number of preprocessing worker (load and preprocess each image)
        :param max_queue_len: (int) Max number of minibatches that can be preprocessed at the same time
        :param infinite_loop: (bool) whether to have an iterator that can be resetted
        :param generative_model_state_dim:([np.array]) The dimension of the latent variable in generative model  
        :param actions : ([np.array]) The actions used to generate observations with Conditional Model
        :param seed :(int) random seed used to generated latent variable
        """
        super(DataLoaderConditional, self).__init__()
        self.minibatchlist = minibatchlist
        self.seed = seed
        self.task = task
        self.TARGET_MAX_X = TARGET_MAX_X
        self.TARGET_MIN_X = TARGET_MIN_X
        self.TARGET_MAX_Y = TARGET_MAX_Y
        self.TARGET_MIN_Y = TARGET_MIN_Y
        
        self.MAX_X = MAX_X
        self.MIN_X = MIN_X
        self.MAX_Y = MAX_Y
        self.MIN_Y = MIN_Y
        
        # For grid walker
        self.grid_walker = grid_walker
        self.grid_list =grid_list
        self.target_pos_list = target_pos_list
        
        self.actions = actions
        self.generative_model_state_dim = generative_model_state_dim
        self.queue = Queue(max_queue_len)
        self.infinite_loop = infinite_loop
        self.process = None
        self.startProcess()
        

    @staticmethod
    def createTestMinibatchList(n_samples, batch_size):
        """
        Create list of minibatch for dataloader
        :param n_samples: (int)
        :param batch_size: (int)
        :return: ([np.array])
        """
        minibatchlist = []
        if n_samples%batch_size == 0:
            n_batchs = n_samples // batch_size
        else:
            n_batchs = n_samples // batch_size +1
        for i in range(n_batchs):
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
        while start or self.infinite_loop:
            start = False    
            indices = np.arange(len(self.minibatchlist), dtype=np.int64)
            for minibatch_idx in indices:
                batch = self._makeBatchElement(minibatch_idx, self.minibatchlist, self.generative_model_state_dim, self.actions,self.task, self.seed, self.TARGET_MAX_X, self.TARGET_MIN_X,self.TARGET_MAX_Y, self.TARGET_MIN_Y,self.MAX_X, self.MIN_X,self.MAX_Y, self.MIN_Y, self.grid_walker, self.grid_list, self.target_pos_list)
                self.queue.put(batch)

                # Free memory
                del batch

            self.queue.put(None)

    @classmethod
    def _makeBatchElement(cls, minibatch_idx, minibatchlist, generative_model_state_dim, actions,task, seed, TARGET_MAX_X, TARGET_MIN_X,TARGET_MAX_Y, TARGET_MIN_Y,MAX_X, MIN_X, MAX_Y, MIN_Y, grid_walker, grid_list, target_pos_list ):
        """
        """
        np.random.seed(seed+minibatch_idx)
        z = th.from_numpy(np.random.normal(0,1,(minibatchlist[minibatch_idx].shape[0], generative_model_state_dim))).float()
        ## Add a fix z instead of random z
       #  z = []
       #  z_0 = [-0.9669311 , -1.7317313 , -0.05010746,  0.43163386,  0.57693458,
       #  0.81835371, -2.35364032, -1.00514448,  0.1066523 ,  1.51900327,
       #  0.78374451,  1.90134001, -0.52493942,  0.27441698, -1.09997082,
       # -0.40435222, -0.73529571, -0.63398868, -0.39344913,  0.00271754,
       #  0.02221266,  0.54345345,  0.13998847, -0.34404564, -0.52257854,
       # -0.30713171, -0.44903713,  0.49097106,  0.86552519,  1.27404451,
       # -0.79770279,  0.46937221, -1.39467967,  0.37317473,  1.08267224,
       # -0.14958951,  1.07263601, -1.13856792, -0.88864529, -0.13580984,
       #  1.02221036, -0.41742945, -0.45355311, -0.99162835,  0.20288104,
       #  1.24669516,  0.70068014,  0.69665068, -0.20697448, -0.56330937,
       #  0.67724591, -0.03191108, -0.17360823,  0.89824063, -0.19778745,
       # -0.83777624,  0.90918851,  0.08071989, -1.03702939, -1.11290586,
       #  0.09541187,  2.33740973, -0.3928206 , -0.33627385,  1.52377117,
       # -0.0572812 , -1.4484669 , -1.57279646,  1.22666395,  0.66635454,
       #  0.82612568, -0.05775656, -0.72671205, -0.21716312,  0.13603121,
       # -0.83831114,  0.56144989, -1.25959611, -0.33275875, -0.20400788,
       # -0.69101983, -2.20550537,  0.44786966, -0.75575078,  1.32570791,
       # -0.34198228, -0.5413596 ,  0.09152195,  1.05343974, -0.56340766,
       #  1.01473773,  1.44030368,  0.99032283,  1.62643147,  1.29264605,
       #  1.51488233,  1.60432637,  0.20806953, -0.4292239 , -2.26224375,
       # -1.32273316, -0.44828281, -0.38173509, -0.15279447, -1.00076044,
       # -1.59577763, -0.13022317, -0.18941793, -0.80755407, -0.74215215,
       # -0.94015658, -0.39652374, -0.8563028 ,  1.2598753 ,  0.24099673,
       # -0.97231793, -0.28044778, -1.18028557,  1.01216829,  1.38418674,
       #  1.252002  , -1.14469266, -0.09126702, -0.40157068,  0.56201309,
       # -1.00790977, -0.6758917 , -0.41321704,  0.15328847,  0.69412869,
       # -0.32872769,  0.66396505,  0.82207638, -0.21321523, -1.24565816,
       # -1.17119038,  0.59172696, -0.47622442, -1.71262932,  0.61295235,
       #  0.12955453, -1.40596712,  1.17941999,  0.83663601,  0.13874525,
       # -1.27431941, -1.40233052, -0.3070685 , -1.71391535,  0.40508026,
       # -1.41082335,  0.16491273, -0.28813145,  0.71178526, -0.93794757,
       #  0.27372944, -1.39484024,  0.79554957, -0.11496177,  0.49585068,
       # -1.32052529,  0.49908426,  0.3062034 ,  0.36369789,  0.31263396,
       # -0.19346388,  1.24129927, -0.15589799, -0.73916918, -0.05872619,
       # -0.95051795, -0.46399641, -0.17724662, -0.37955412,  0.19939707,
       #  1.94576144,  0.57094985,  1.07230067, -0.50370944, -0.58701628,
       # -0.37817806,  0.85288912, -2.1481185 , -1.03316474,  0.10233585,
       # -0.22409236,  1.96772969,  0.44768322, -0.66219145, -1.57760704,
       # -0.34056005, -1.30322003,  0.46675065,  0.16110632,  0.32003194,
       #  2.07917666, -0.90746599, -0.19240421, -1.21251571, -0.08059852]
       #  for _ in range(minibatchlist[minibatch_idx].shape[0]):
       #      z.append(z_0)
       #  z = th.FloatTensor(z)
       
        
        if not grid_walker:
            actions = th.FloatTensor(actions[minibatchlist[minibatch_idx]])
            if task == 'sc':
                random_init_x = np.random.random_sample(minibatchlist[minibatch_idx].shape[0]) * (TARGET_MAX_X - TARGET_MIN_X) + \
                            TARGET_MIN_X
                random_init_y = np.random.random_sample(minibatchlist[minibatch_idx].shape[0]) * (TARGET_MAX_Y - TARGET_MIN_Y) + \
                            TARGET_MIN_Y
            else:
                random_init_x = np.zeros(minibatchlist[minibatch_idx].shape[0])
                random_init_y = np.zeros(minibatchlist[minibatch_idx].shape[0])
            init_robot_x = np.random.random_sample(minibatchlist[minibatch_idx].shape[0]) * (MAX_X - MIN_X) + \
                            MIN_X
            init_robot_y = np.random.random_sample(minibatchlist[minibatch_idx].shape[0]) * (MAX_Y - MIN_Y) + \
                        MIN_Y
            
        
            target_pos = th.FloatTensor(np.concatenate((random_init_x[...,None], random_init_y[...,None]),axis=1))
            robot_pos = th.FloatTensor(np.concatenate((init_robot_x[...,None], init_robot_y[...,None]),axis=1))
            return (z, actions, target_pos, robot_pos)
        else:
            target_pos = th.FloatTensor(target_pos_list[minibatchlist[minibatch_idx]])
            robot_pos = th.FloatTensor(grid_list[minibatchlist[minibatch_idx]])
            return (z, th.zeros(1), target_pos, robot_pos) 
        
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

    def __init__(self, x_indices, y_values, images_path, batch_size, n_workers=1, no_targets=False,
                 shuffle=False, infinite_loop=True, max_queue_len=4, absolute_path=False):
        # Create minibatch list
        minibatchlist, targets = self.createMinibatchList(x_indices, y_values, batch_size)

        # Whether to yield targets together with output
        # (not needed when plotting or predicting states)
        self.no_targets = no_targets
        self.targets = np.array(targets)
        self.shuffle = shuffle
        self.absolute_path = absolute_path
        super(SupervisedDataLoader, self).__init__(minibatchlist, images_path, n_workers=n_workers,
                                                   infinite_loop=infinite_loop, max_queue_len=max_queue_len,
                                                                  absolute_path=self.absolute_path)

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
                        batch = [self._makeBatchElement(image_path,
                                                        absolute_path=self.absolute_path) for image_path in images]
                    else:
                        batch = parallel(delayed(self._makeBatchElement)(image_path) for image_path in images)

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
    r"""Robot Envinronment Dataset for DataLoader (Pytorch natively supported)

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
    :param mode: (int)
    :param img_shape: (tuple or None) if None, image will not be resize, else: resize image to new shape (channels first)) e.g. img_shape = (3, 128, 128).

        Set to True, the dataloader will output both `obs` and `next_obs` (a tuple of th.Tensor)
        Set to false, it will only output one th.Tensor.
    """

    def __init__(self, sample_indices, images_path, actions, all_imgs_target_pos,relative_positions, rewards, episode_starts,
                 img_shape=None,
                 mode=1,
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
        self.all_imgs_target_pos = all_imgs_target_pos
        self.relative_positions = relative_positions
        self.actions = actions
        self.rewards = rewards
        self.episode_starts = episode_starts

        self.img_shape = img_shape
        self.mode = mode
        self.use_triplets = use_triplets
        self.multi_view = multi_view
        # apply occlusion for training a DAE
        self.apply_occlusion = apply_occlusion
        self.occlusion_percentage = occlusion_percentage

        self.dtype = dtype
        self.img_extension = img_extension
        
        self.class_labels = np.array(list(map(lambda x:int(x.split("/")[-2].split("_")[-1]), images_path)))
        # self.random_target_balance = random_target_balance
        
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
            if self.mode == 1:
                img = self._get_one_img(image_path)
                img_next = self._get_one_img(self.images_path[index+1])
                action = self.actions[index]
                next_action = self.actions[index+1]
                target_pos = self.all_imgs_target_pos[index]
                next_target_pos = self.all_imgs_target_pos[index+1]
                robot_pos = self.relative_positions[index]
                next_robot_pos = self.relative_positions[index+1]
                reward = self.rewards[index]
                cls_gt = self.class_labels[index]
                return index, img.astype(self.dtype), img_next.astype(self.dtype), action, next_action, reward, cls_gt, target_pos, next_target_pos, robot_pos, next_robot_pos
            elif self.mode == 0:
                img = self._get_one_img(image_path)
                action = self.actions[index]
                target_pos = self.all_imgs_target_pos[index]
                robot_pos = self.relative_positions[index]
                return img.astype(self.dtype), action, target_pos,  robot_pos
            elif self.mode == 2:
                img = self._get_one_img(image_path)
                img_next = self._get_one_img(self.images_path[index+1])
                reward = self.rewards[index]
                action = self.actions[index]
                next_action = self.actions[index+1]
                target_pos = self.all_imgs_target_pos[index]
                next_target_pos = self.all_imgs_target_pos[index+1]
                robot_pos = self.relative_positions[index]
                next_robot_pos = self.relative_positions[index+1]
                
                return img.astype(self.dtype), img_next.astype(self.dtype), reward,action, next_action, target_pos, next_target_pos, robot_pos, next_robot_pos  
            else:
                return

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

