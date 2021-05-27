
import os
from os.path import dirname, abspath
import itertools
import imageio
from tqdm import tqdm

import torch as th
import torch.utils.data

from src.model import SubTab
from utils.load_data import Loader
from sklearn.preprocessing import StandardScaler
from utils.arguments import print_config_summary
from utils.arguments import get_arguments, get_config
from utils.utils import set_dirs, run_with_profiler, update_config_with_model_dims, tsne
from utils.eval_utils import linear_model_eval, plot_clusters, append_tensors_to_lists, concatenate_lists

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import mlflow

torch.manual_seed(1)


def eval(data_loader, config):
    """
    :param IterableDataset data_loader: Pytorch data loader.
    :param dict config: Dictionary containing options.
    :return: None
    """
    # Instantiate Autoencoder model
    model = SubTab(config)
    # Load the model
    model.load_models()
    # Evaluate Autoencoder
    with th.no_grad():
        x_train_hat, y_train = evalulate_models(data_loader, model, config, plot_suffix="train", mode="train")
        evalulate_models(data_loader, model, config, plot_suffix="test", mode="test", x_train_hat=x_train_hat, y_train=y_train)
        print(f"Results are saved under ./results/{config['framework']}/evaluation/clusters/")


def evalulate_models(data_loader, model, config, plot_suffix="_Test", mode='train', x_train_hat=None, y_train=None):
    """
    :param IterableDataset data_loader: Pytorch data loader.
    :param model: Pre-trained autoencoder class.
    :param dict config: Dictionary containing options.
    :param plot_suffix: Custom suffix to use when saving plots.
    :return: None.
    """
    # Print whether we are evaluating training set, or test set
    print(f"{100 * '#'}\n{100 * '#'}")
    print(f"Evaluating on " + plot_suffix + " set...")
    # Print domain names
    print(f"{100 * '='}\n{100 * '='}")
    print(f"Dataset used: {config['dataset']}")
    print(f"{100 * '='}\n{100 * '='}")

    # Get Autoencoders for both modalities
    encoder = model.encoder
    # Move the models to the device
    encoder.to(config["device"])
    # Set models to evaluation mode
    encoder.eval()

    # Get data loaders for three datasets
    train_loader, test_loader, validation_loader = data_loader.train_loader, data_loader.test_loader, data_loader.validation_loader
    # Choose either training, or test data loader
    if mode == 'train':
        data_loader_tr_or_te = train_loader 
    elif mode == 'test':
        data_loader_tr_or_te = test_loader
    else:
        print("Error with data loading...")
        
    # Attach progress bar to data_loader to check it during training. "leave=True" gives a new line per epoch
    train_tqdm = tqdm(enumerate(data_loader_tr_or_te), total=len(data_loader_tr_or_te), leave=True)

    # Create empty lists to hold data.
    Xd_l, clabels_l = [], []
    Xr_l, z_l = [], []

    # Go through batches
    for i, (x, label) in train_tqdm:

        #  Concatenate original data with itself to be used when computing reconstruction error w.r.t reconstructions from xi and xj
        Xorig = model.process_batch(x, x)

        # Generate corrupted samples
        x_tilde_list, labels_list = model.subset_generator(x)
                
        # zip features and labels 
        feature_label_pairs = zip(x_tilde_list, labels_list)
        
        # Get number of subsets, and compute their combination [((xi1, yi1), (xj1, yj1)), ((xi1, yi1), (xj2, yj2))...]
        subset_combinations = list(itertools.combinations(feature_label_pairs, 2))
                
        latent_list = []    
        # Concatenate xi, and xj, and turn it into a tensor
        for ((xi, yi), (xj, yj)) in subset_combinations:
            Xbatch = model.process_batch(xi, xj)
            z, latent, recon_image = encoder(Xbatch)
            latent = split_and_average(latent, config["batch_size"])
            latent_list.append(latent)
            
        # Average latents from all combinations
        latent = sum(latent_list)/len(latent_list)
        # Get the labels
        label_idx = label.int()
        # Append tensors to the corresponding lists as numpy arrays
        Xd_l, Xr_l, z_l, clabels_l = append_tensors_to_lists([Xd_l, Xr_l, z_l, clabels_l],
                                                             [x, recon_image, latent, label_idx])

    # Turn list of numpy arrays to a single numpy array for input data, reconstruction, and latent samples.
    Xdata, Xrecon, z = concatenate_lists([Xd_l, Xr_l, z_l])
    # Turn list of numpy arrays to a single numpy array for cohort and domain labels (labels such as 1, 2, 3, 4,...).
    clabels = concatenate_lists([clabels_l])

    # Visualise clusters
    plot_clusters(config, z, clabels[0],  plot_suffix="_inLatentSpace_" + plot_suffix)

    if mode == 'test':
        # Classification scores using Logistig Regression
        print(20 * "*" + " Classification scores using embeddings " + 20 * "*")
        x_test_hat = z
        y_test = clabels[0]
        linear_model_eval(config, x_train_hat, y_train, x_test_hat, y_test,
                          description="Results for Logistic Reg. trained on the representations of training set and tested on that of test set:")

    x_train_hat = z
    y_train = clabels[0]
    return x_train_hat, y_train


def split_and_average(x, batch_size):
    xi, xj = th.split(x, batch_size)
    return (xi+xj)/2.0


def main(config):
    # Ser directories (or create if they don't exist)
    set_dirs(config)
    # Get data loader for first dataset.
    ds_loader = Loader(config, dataset_name=config["dataset"])
    # Add the number of features in a dataset as the first dimension of the model
    config = update_config_with_model_dims(ds_loader, config)
    # Start evaluation
    eval(ds_loader, config)


if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    config = get_config(args)
    # Get all of available training set for evaluation (i.e. no need for validation set)
    config["training_data_ratio"] = 1.0
    # Increase batch size to speed up loading data
    config["batch_size"] = 1000
    # Turn off noise when evaluating the performance
    config["add_noise"] = False
    # Summarize config and arguments on the screen as a sanity check
    print_config_summary(config, args)
    # --If True, start of MLFlow for experiment tracking:
    if config["mlflow"]:
        # Experiment name
        mlflow.set_experiment(experiment_name=config["model_mode"] + "_" + str(args.experiment))
        # Start a new mlflow run
        with mlflow.start_run():
            # Run the main with or without profiler
            run_with_profiler(main, config) if config["profile"] else main(config)
    else:
        # Run the main with or without profiler
        run_with_profiler(main, config) if config["profile"] else main(config)
