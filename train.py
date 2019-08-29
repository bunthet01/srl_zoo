"""
Part of this program is based on the implementation by Rico Jonschkowski (rico.jonschkowski@tu-berlin.de):
https://github.com/tu-rbo/learning-state-representations-with-robotic-priors

"""
from __future__ import print_function, division, absolute_import

import argparse
from collections import OrderedDict
import os
import numpy as np
import torch as th

import preprocessing
import models.learner as learner
import plotting.representation_plot as plot_script
from models.learner import SRL4robotics
from pipeline import getLogFolderName, saveConfig, correlationCall
from plotting.losses_plot import plotLosses, plotloss_G_D
from plotting.representation_plot import plotRepresentation
from utils import parseDataFolder, createFolder, getInputBuiltin, loadData, buildConfig, parseLossArguments
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='State Representation Learning with PyTorch')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--img-shape', type=str, default="(3,64,64)",
                        help='image shape (default "(3,64,64)"')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--state-dim', type=int, default=2, help='state dimension (default: 2)')
    parser.add_argument('-bs', '--batch-size', type=int, default=32, help='batch_size (default: 32)')
    parser.add_argument('--val-size', type=float, default=0.2, help='Validation set size in percentage (default: 0.2)')
    parser.add_argument('--training-set-size', type=int, default=-1,
                        help='Limit size (number of samples) of the training set (default: -1)')			# for now, we use the default value, there might be an error when use smaller size
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0005, help='learning rate (default: 0.005)')
    parser.add_argument('-lr_G', '--learning-rate-G', type=float, default=1.0 *
                        1e-5, help='learning rate GAN: Generator (default: None)')
    parser.add_argument('-lr_D', '--learning-rate-D', type=float, default=5.0*1e-5,
                        help='learning rate GAN: Discriminator (default: None)')
    parser.add_argument('--l1-reg', type=float, default=0.0, help='L1 regularization coeff (default: 0.0)')
    parser.add_argument('--l2-reg', type=float, default=0.0, help='L2 regularization coeff (default: 0.0)')
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-num', type=int, default=0, help='CUDA visible device (use CPU if -1, default: 0)')
    parser.add_argument('--no-display-plots', action='store_true', default=False,
                        help='disables live plots of the representation learned')
    parser.add_argument('--model-type', type=str, default="custom_cnn",
                        choices=['custom_cnn', 'custom_cnn_2', 'resnet', 'mlp', 'linear', 'gan', 'unet','dc'],
                        help='Model architecture (default: "custom_cnn")')
    parser.add_argument('--inverse-model-type', type=str, default="linear",
                        choices=['mlp', 'linear'],
                        help='Inverse model s architecture (default: "linear")')
    parser.add_argument('--data-folder', type=str, default="", help='Dataset folder', required=True)
    parser.add_argument('--log-folder', type=str, default="",
                        help='Folder where the experiment model and plots will be saved. ' +
                             'By default, automatically computing KNN-MSE and saving logs at location ' +
                             'logs/DatasetName/YY-MM-DD_HHhMM_SS_ModelType_ST_DIMN_LOSSES')
    parser.add_argument('--multi-view', action='store_true', default=False,
                        help='Enable use of multiple camera')
    parser.add_argument('--balanced-sampling', action='store_true', default=False,
                        help='Force balanced sampling for episode independent prior instead of uniform')
    parser.add_argument('--losses', nargs='+', default=[], **parseLossArguments(
        choices=["forward", "inverse", "reward", "reward2", "spcls", "priors", "episode-prior", "reward-prior", "triplet",
                 "autoencoder", "vae","cvae", "perceptual", "dae", "random", "gan","cgan", "gan_new", "cgan_new", "cvae_new"],
        help='The wanted losses. One may also want to specify a weight and dimension '
             'that apply as follows: "<name>:<weight>:<dimension>".'))
    parser.add_argument('--beta', type=float, default=1.0,
                        help='(For beta-VAE only) Factor on the KL divergence, higher value means more disentangling.')
    parser.add_argument('--path-to-dae', type=str, default="",
                        help='Path to a pre-trained dae model when using the perceptual loss with VAE')
    parser.add_argument('--state-dim-dae', type=int, default=200,
                        help='state dimension of the pre-trained dae (default: 200)')
    parser.add_argument('--occlusion-percentage', type=float, default=0.5,
                        help='Max percentage of input occlusion for masks when using DAE')
    parser.add_argument('--figdir', type=str, default=None,
                        help="Save figure the 'figdir'.")
    parser.add_argument('--monitor', type=str, default='loss',
                        choices=['pbar', 'loss'],
                        help="Monitor mode: either print the losses ('loss') or show the progressbar ('pbar'). (default 'loss')")
    parser.add_argument('--num-worker', type=int, default=10,
                        help="Number of CPUs to use for dataloader.")
    parser.add_argument('--srl-pre-weights', type=str, default=None,
                        help="Load SRL pretrained weights.")
    parser.add_argument('--debug', action='store_true', default=False,
                        help="Debug mode.")  
    parser.add_argument('--Cmax', type=float, default=50,
                        help='(For CCI-VAE or CCI-CVAE)') 
    parser.add_argument('--use-cci', action='store_true', default=False, help='if cci is use in VAE')
    parser.add_argument('--ls', action='store_true', default=False, help='used to smooth the real label in dcGAN and CdcGAN')
    parser.add_argument('--add-noise', action='store_true', default=False, help='Add noise to the input of the discriminator of dcGAN and CdcGan ')
    parser.add_argument('--only-action', action='store_true', default=False, help='Conditioned only on action')

                     
    args = parser.parse_args()
    # args.cuda = not args.no_cuda and th.cuda.is_available()
    args.data_folder = parseDataFolder(args.data_folder)
    learner.SAVE_PLOTS = not args.no_display_plots
    learner.N_EPOCHS = args.epochs
    learner.BATCH_SIZE = args.batch_size
    learner.VALIDATION_SIZE = args.val_size
    learner.BALANCED_SAMPLING = args.balanced_sampling
    learner.N_WORKERS = args.num_worker
    th.backends.cudnn.benchmark = True
    # Dealing with losses to use
    has_loss_description = [isinstance(loss, tuple) for loss in args.losses]
    has_consistent_description, has_weight, has_splits = False, False, False
    if all(has_loss_description) and len(has_loss_description) > 0:
        len_description = [len(item_loss) for item_loss in args.losses]
        has_consistent_description = sum(len_description) / len(len_description) == len_description[0]
        has_weight = has_consistent_description
        has_splits = has_consistent_description and len_description[0] == 3

    if any(has_loss_description) and not all(has_loss_description):
        raise ValueError(
            "Either no losses have a defined weight or dimension, or all losses have a defined weight. {}".format(args.losses))

    # If not describing the the losses (weight and or dimension)
    if not has_consistent_description:
        losses = list(set(args.losses))
        losses_weights_dict = None
        split_dimensions = -1

    # otherwise collecting descriptions
    else:
        losses_weights_dict = OrderedDict()
        split_dimensions = OrderedDict()
        for loss, weight, split_dim in args.losses:
            losses_weights_dict[loss] = weight
            split_dimensions[loss] = split_dim
        losses = list(losses_weights_dict.keys())

        if not has_weight:
            split_dimensions = losses_weights_dict
            losses_weights_dict = None
            losses = list(split_dimensions.keys())

        assert not ("triplet" in losses and not args.multi_view), \
            "Triplet loss with single view is not supported, please use the --multi-view option"
    args.losses = losses
    args.split_dimensions = split_dimensions
    if args.img_shape is None:
        img_shape = None  # (3,224,224)
    else:
        img_shape = tuple(map(int, args.img_shape[1:-1].split(",")))
    if args.multi_view is True:
        # Setting variables involved data-loading from multiple cameras,
        # involved also in adapting the input layers of NN to that data
        # PS: those are stacked images - 3 if triplet loss, 2 otherwise
        if "triplet" in losses:
            # preprocessing.preprocess.N_CHANNELS = 9
            img_shape = (9, ) + img_shape[1:]
        else:
            # preprocessing.preprocess.N_CHANNELS = 6
            img_shape = (6, ) + img_shape[1:]

    assert not ("autoencoder" in losses and "vae" in losses), "Model cannot be both an Autoencoder and a VAE (come on!)"
    assert not (("autoencoder" in losses or "vae" in losses)
                and args.model_type == "resnet"), "Model cannot be an Autoencoder or VAE using ResNet Architecture !"
    assert not ("vae" in losses and args.model_type == "linear"), "Model cannot be VAE using Linear Architecture !"
    assert not (args.multi_view and args.model_type == "resnet"), \
        "Default ResNet input layer is not suitable for stacked images!"
    assert not (args.path_to_dae == "" and "vae" in losses and "perceptual" in losses), \
        "To use the perceptual loss with a VAE, please specify a path to a pre-trained DAE model"
    assert not ("dae" in losses and "perceptual" in losses), \
        "Please learn the DAE before learning a VAE with the perceptual loss "
    assert not (args.use_cci and ("vae" not in losses and "cvae" not in losses)), "cci cannot used without vae or cvae"
    assert not ("cvae" in losses and "cvae_new" in losses), "can not cvae and cvae_new at the same time"
    assert not ("cgan" in losses and len(losses)>1), "can not use cgan with other models"
    assert not ("cgan_new" in losses and len(losses)>1), "can not use cgan_new with other models"
    

    print('Loading data ... ')

    training_data, ground_truth, relative_positions, target_positions = loadData(args.data_folder)
    rewards, episode_starts = training_data['rewards'], training_data['episode_starts']
    actions = training_data['actions']
    # We assume actions are integers
    n_actions = int(np.max(actions) + 1)

    # Try to convert old python 2 format
    try:
        images_path = np.array([path.decode("utf-8") for path in ground_truth['images_path']])
    except AttributeError:
        images_path = ground_truth['images_path']

    # Building the experiment config file
    exp_config = buildConfig(args)

    if args.log_folder == "":
        # Automatically create dated log folder for configs
        createFolder("logs/{}".format(exp_config['data-folder']), "Dataset log folder already exist")
        # Check that the dataset is already preprocessed
        log_folder, experiment_name = getLogFolderName(exp_config)
        args.log_folder = log_folder
    else:
        os.makedirs(args.log_folder, exist_ok=True)
        experiment_name = "{}_{}".format(args.model_type, losses)

    exp_config['log-folder'] = args.log_folder
    exp_config['experiment-name'] = experiment_name
    exp_config['n_actions'] = n_actions
    exp_config['multi-view'] = args.multi_view
    exp_config['img_shape'] = args.img_shape
    exp_config['use_cci'] = args.use_cci
    exp_config['Cmax'] = args.Cmax
    exp_config['learning_rate_D'] = args.learning_rate_D
    exp_config['learning_rate_G'] = args.learning_rate_G
    exp_config['label_smoothing'] = args.ls
    exp_config['add_noise'] = args.add_noise
    exp_config['only_action'] = args.only_action
    exp_config['debug'] = args.debug
    exp_config['pretrained_weights_path'] = args.srl_pre_weights
    

    if "dae" in losses:
        exp_config['occlusion-percentage'] = args.occlusion_percentage
    print('Log folder: {}'.format(args.log_folder))

    print('Learning a state representation ... ')
    # The dimension of the class (action) in CVAE
    class_dim = 4
    exp_config['class_dim'] = class_dim

    srl = SRL4robotics(args.state_dim, class_dim, img_shape=img_shape, model_type=args.model_type, inverse_model_type=args.inverse_model_type
                        , ls=args.ls,add_noise=args.add_noise,only_action = args.only_action, pretrained_weights_path=args.srl_pre_weights,debug=args.debug,
                       seed=args.seed,log_folder=args.log_folder, learning_rate=args.learning_rate, learning_rate_gan=( args.learning_rate_D,
                        args.learning_rate_G),l1_reg=args.l1_reg, l2_reg=args.l2_reg, cuda=args.gpu_num, multi_view=args.multi_view,losses=losses,
                        losses_weights_dict=losses_weights_dict, n_actions=n_actions, beta=args.beta, use_cci=args.use_cci, Cmax=args.Cmax,
                        split_dimensions=split_dimensions, path_to_dae=args.path_to_dae, state_dim_dae=args.state_dim_dae, 
                        occlusion_percentage=args.occlusion_percentage)

    if args.training_set_size > 0:
        limit = args.training_set_size
        actions = actions[:limit]
        images_path = images_path[:limit]
        rewards = rewards[:limit]
        episode_starts = episode_starts[:limit]
        truncate = limit
    else:
        truncate = None 
    # Save configs in log folder
    saveConfig(exp_config, print_config=True)
    if args.figdir is not None:
        os.makedirs(args.figdir, exist_ok=True)
    loss_history, learned_states, pairs_name_weights = srl.learn(
        images_path, actions, rewards, episode_starts, figdir=args.figdir, monitor_mode=args.monitor,
        ground_truth=ground_truth, relative_positions=relative_positions, target_positions=target_positions, truncate=truncate)

    # Update config with weights for each losses
    exp_config['losses_weights'] = pairs_name_weights
    saveConfig(exp_config, print_config=True)

    if learned_states is not None:
        print("learned_states.shape ", learned_states.shape)
        srl.saveStates(learned_states, images_path, rewards, args.log_folder)
        name = "Learned State Representation\n {}".format(args.log_folder.split('/')[-1])
        path = "{}/learned_states.png".format(args.log_folder)

        # PLOT REPRESENTATION & CORRELATION
        plotRepresentation(learned_states, rewards, name, add_colorbar=True, path=path)
        correlationCall(exp_config, plot=not args.no_display_plots)

    if "gan" in losses or "cgan" in losses:
        # Save plot
        plotLosses(loss_history[0], path=args.log_folder, name="Autoencoder_losses")       
        plotLosses(loss_history[1], path=args.log_folder, name="Dicriminator_losses")
        plotLosses(loss_history[2], path=args.log_folder, name="Generator_losses")
        plotloss_G_D(loss_history[2], loss_history[1], path=args.log_folder, name="G_D_losses")

    
        loss_history = {**loss_history[0],**loss_history[1],**loss_history[2]} 
    elif "gan_new" in losses or "cgan_new" in losses:
        # Save plot    
        plotLosses(loss_history[0], path=args.log_folder, name="Dicriminator_losses")
        plotLosses(loss_history[1], path=args.log_folder, name="Generator_losses")
        plotloss_G_D(loss_history[1], loss_history[0], path=args.log_folder, name="G_D_losses")
	
    
        loss_history = {**loss_history[0],**loss_history[1]} 
    else:
        plotLosses(loss_history, args.log_folder)
    np.savez('{}/loss_history.npz'.format(args.log_folder), **loss_history)
    

    
