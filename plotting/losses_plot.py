from __future__ import print_function, division

import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Init seaborn
sns.set()


def plotLosses(loss_history, path=None, name="Losses"):
    """
    :param loss_history: (dict)
    :param path: (str)
    """
    keys = list(loss_history.keys())
    keys.sort()
    plt.figure(name)
    for key in keys:
        # check if the loss was averaged by epoch or not yet
        if isinstance(loss_history[key], np.ndarray) and len(loss_history[key].shape) > 1:
            # the axis here means every axis but the first, so average over every dim, except dim 0
            loss_hist_key = np.mean(loss_history[key], axis=tuple(range(1, len(loss_history[key].shape))))
        else:
            loss_hist_key = loss_history[key]
        x = np.arange(1, len(loss_hist_key) + 1)
        plt.plot(x, loss_hist_key, label=key, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title("Evolution of the losses")
    if path is not None:
        plt.savefig(path +"/"+ name+".png")
    else:
        plt.show()
        
def plotloss_G_D(loss_history_G, loss_history_D, path=None, name='Losses_G_D'):
    keys_D = list(loss_history_D.keys())
    keys_G = list(loss_history_G.keys())
    plt.figure(name)
    for key in keys_D:
        if key == 'train_loss_D':
            # check if the loss was averaged by epoch or not yet
            if isinstance(loss_history_D[key], np.ndarray) and len(loss_history_D[key].shape) > 1:
                # the axis here means every axis but the first, so average over every dim, except dim 0
                loss_hist_key = np.mean(loss_history_D[key], axis=tuple(range(1, len(loss_history_D[key].shape))))
            else:
                loss_hist_key = loss_history_D[key]
            x = np.arange(1, len(loss_hist_key) + 1)
            plt.plot(x, loss_hist_key, label=key, linewidth=2)
            
    for key in keys_G:
        if key == 'train_loss_G':
            # check if the loss was averaged by epoch or not yet
            if isinstance(loss_history_G[key], np.ndarray) and len(loss_history_G[key].shape) > 1:
                # the axis here means every axis but the first, so average over every dim, except dim 0
                loss_hist_key = np.mean(loss_history_G[key], axis=tuple(range(1, len(loss_history_G[key].shape))))
            else:
                loss_hist_key = loss_history_G[key]
            x = np.arange(1, len(loss_hist_key) + 1)
            plt.plot(x, loss_hist_key, label=key, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title("Evolution of the losses")
    if path is not None:
        plt.savefig(path +"/"+ name+".png")
    else:
        plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot losses')
    parser.add_argument('-i', '--input-file', type=str, default="", required=True,
                        help='Path to a npz file containing losses history')
    parser.add_argument('--log-folder', type=str, default="", help='Path to a log folder')
    args = parser.parse_args()

    path = args.log_folder if args.log_folder != "" else None

    plotLosses(np.load(args.input_file), path)
