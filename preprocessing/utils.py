from __future__ import print_function, division

import numpy as np
import torch as th
import re


def preprocessInput(x, mode="tf"):
    """
    Normalize input
    :param x: (np.ndarray) (RGB image with values between [0, 255])
    :param mode: (str) One of "image_net", "tf".
        - image_net: will zero-center each color channel with
            respect to the ImageNet dataset,
            with scaling.
            cf http://pytorch.org/docs/master/torchvision/models.html
        - tf: will scale pixels between -1 and 1,
            sample-wise.
    :return: (np.ndarray)
    """
    assert x.shape[-1] == 3, "Color channel must be at the end of the tensor {}".format(x.shape)
    x /= 255.
    if mode == "tf":
        x -= 0.5
        x *= 2.
    elif mode == "image_net":
        # Zero-center by mean pixel
        x[..., 0] -= 0.485
        x[..., 1] -= 0.456
        x[..., 2] -= 0.406
        # Scaling
        x[..., 0] /= 0.229
        x[..., 1] /= 0.224
        x[..., 2] /= 0.225

    else:
        raise ValueError("Unknown mode for preprocessing")
    return x


def deNormalize(x, mode="tf"):
    """
    deNormalize data (transform input to [0, 1])
    :param x: (np.ndarray)
    :param mode: (str) One of "image_net", "tf".
    :return: (np.ndarray)
    """
    # Reorder channels when we have only one image
    if x.shape[0] == 3 and len(x.shape) == 3:
        # (n_channels, height, width) -> (height, width, n_channels)
        x = np.transpose(x, (1, 2, 0))

    assert x.shape[-1] == 3, "Color channel must be at the end of the tensor {}".format(x.shape)

    if mode == "tf":
        x /= 2.
        x += 0.5
    elif mode == "image_net":
        # Scaling
        x[..., 0] *= 0.229
        x[..., 1] *= 0.224
        x[..., 2] *= 0.225
        # Undo Zero-center
        x[..., 0] += 0.485
        x[..., 1] += 0.456
        x[..., 2] += 0.406
    else:
        raise ValueError("Unknown mode for deNormalize")
    # Clip to fix numeric imprecision (1e-09 = 0)
    return np.clip(x, 0, 1)

def one_hot(x):
    """
    Convert action to onehot tensor. (used for cvae.py)
    :param x: th.tensor() or np.ndarray() or list
    :return: th.FloatTensor()
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
    return th.FloatTensor(result)
    
def gaussian_target(img_shape, t, MAX_X=0.85, MIN_X=-0.85, MAX_Y=0.85, MIN_Y=-0.85, sigma2=10):
    """
    Create a gaussian bivariate tensor for target or robot position.
    :param t: (th.Tensor) Target position (or robot position)
    """
    X_range = img_shape[1]
    Y_range = img_shape[2]
    XY_range = np.arange(X_range*Y_range)
    for i in range(t.size(0)):
      X_t = int((MAX_X+t[i][1])*(img_shape[1]/(MAX_X-MIN_X)))
      Y_t = int((MAX_Y-t[i][0])*(img_shape[2]/(MAX_Y-MIN_Y)))
      bi_var_gaussian = -0.5 * (((XY_range // X_range)- X_t)**2 + (XY_range - (XY_range//Y_range)*Y_range - Y_t)**2)/sigma2
      img_target = th.from_numpy((np.exp(bi_var_gaussian)/(2*np.pi*sigma2)).reshape(X_range, Y_range))
      img_target = img_target[None,...][None,...]
      if i==0: output = img_target
      else: output = th.cat([output,img_target],0)
    return output

def attach_target_pos_to_all_imgs(images_path, target_pos):
    """
    The target_positions in ground_truth.npz equal to the number of episode.
    This function will attach the target_position to every frames.
    """
    all_target_pos = np.zeros((images_path.shape[0],2))
    for i in range(images_path.shape[0]):
        x = re.findall("_[0-9][0-9][0-9]", images_path[i])
        x = int(x[0].replace("_",""))
        all_target_pos[i] = target_pos[i]
    return all_target_pos


def sample_target_pos(batch_size,TARGET_MAX_X, TARGET_MIN_X, TARGET_MAX_Y, TARGET_MIN_Y):
    """
    Sample target_position or robot_position by respecting to their limits.
    """
    random_init_x = np.random.random_sample(batch_size) * (TARGET_MAX_X - TARGET_MIN_X) + \
                    TARGET_MIN_X
    random_init_y = np.random.random_sample(batch_size) * (TARGET_MAX_Y - TARGET_MIN_Y) + \
                TARGET_MIN_Y
    return th.FloatTensor(np.concatenate((random_init_x[...,None], random_init_y[...,None]),axis=1))
    
