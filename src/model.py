
import os
import gc

import itertools
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from utils.utils import set_seed, set_dirs
from utils.loss_functions import JointLoss
from utils.model_plot import save_loss_plot
from utils.model_utils import AEWrapper

import torch as th
import torch.nn.functional as F
from torch.autograd import Variable


th.autograd.set_detect_anomaly(True)


class SubTab:
    """
    Model: Trains an Autoencoder with a Projection network, using SubTab framework.
    """

    def __init__(self, options):
        """Class to train an autoencoder model with projection in SubTab framework.

        Args:
            options (dict): Configuration dictionary.
        """
        # Get config
        self.options = options
        # Define which device to use: GPU, or CPU
        self.device = options["device"]
        # Create empty lists and dictionary
        self.model_dict, self.summary = {}, {}
        # Set random seed
        set_seed(self.options)
        # Set paths for results and Initialize some arrays to collect data during training
        self._set_paths()
        # Set directories i.e. create ones that are missing.
        set_dirs(self.options)
        # ------Network---------
        # Instantiate networks
        print("Building the models for training and evaluation in SubTab framework...")
        # Set Autoencoders i.e. setting loss, optimizer, and device assignment (GPU, or CPU)
        self.set_autoencoder()
        # Set scheduler (its use is optional)
        self._set_scheduler()
        # Print out model architecture
        self.print_model_summary()

    def set_autoencoder(self):
        """Sets up the autoencoder model, optimizer, and loss"""
        # Instantiate the model for the text Autoencoder
        self.encoder = AEWrapper(self.options)
        # Add the model and its name to a list to save, and load in the future
        self.model_dict.update({"encoder": self.encoder})
        # Assign autoencoder to a device
        for _, model in self.model_dict.items(): model.to(self.device)
        # Get model parameters
        parameters = [model.parameters() for _, model in self.model_dict.items()]
        # Joint loss including contrastive, reconstruction and distance losses
        self.joint_loss = JointLoss(self.options)
        # Set optimizer for autoencoder
        self.optimizer_ae = self._adam(parameters, lr=self.options["learning_rate"])
        # Add items to summary to be used for reporting later
        self.summary.update({"recon_loss": []})

    def set_parallelism(self, model):
        """NOT USED - Sets up parallelism in training."""
        # If we are using GPU, and if there are multiple GPUs, parallelize training
        if th.cuda.is_available() and th.cuda.device_count() > 1:
            print(th.cuda.device_count(), " GPUs will be used!")
            model = th.nn.DataParallel(model)
        return model

    def fit(self, data_loader):
        """Fits model to the data"""
        
        # Get data loaders for three datasets
        train_loader = data_loader.train_loader
        validation_loader = data_loader.validation_loader

        # Placeholders for record batch losses
        self.loss = {"tloss_b": [], "tloss_e": [], "vloss_e": [], "closs_b": [], "rloss_b": []}

        # Turn on training mode for each model.
        self.set_mode(mode="training")

        # Compute total number of batches per epoch
        self.total_batches = len(train_loader)

        # Start joint training of Autoencoder with Projection network
        for epoch in range(self.options["epochs"]):
            # Attach progress bar to data_loader to check it during training. "leave=True" gives a new line per epoch
            self.train_tqdm = tqdm(enumerate(train_loader), total=self.total_batches, leave=True)

            # Go through batches
            for i, (x, _) in self.train_tqdm:

                # Generate corrupted samples -- labels are not used
                x_tilde_list, labels_list = self.subset_generator(x)
                
                # zip features and labels -- labels are not used
                feature_label_pairs = zip(x_tilde_list, labels_list)
                # Get number of subsets, and compute their combination [((xi1, yi1), (xj1, yj1)), ((xi1, yi1), (xj2, yj2))...]
                subset_combinations = list(itertools.combinations(feature_label_pairs, 2))
                
                # Concatenate xi, and xj, and turn it into a tensor
                feature_label_batch_list = []
                for ((xi, yi), (xj, yj)) in subset_combinations:
                    Xbatch = self.process_batch(xi, xj)
                    Ybatch = self.process_batch(yi, yj)
                    feature_label_batch_list.append((Xbatch, Ybatch))    
                
                #  Concatenate original data with itself to be used when computing reconstruction error w.r.t reconstructions from xi and xj
                Xorig = self.process_batch(x, x)

                # 0 - Update Autoencoder
                self.update_autoencoder(feature_label_batch_list, Xorig)

                # 1 - Update log message using epoch and batch numbers
                self.update_log(epoch, i)
                # 2 - Clean-up for efficient memory use.
                gc.collect()

            # Validate every nth epoch. n=1 by default
            if epoch % self.options["nth_epoch"] == 0 and self.options["validate"]:
                # Compute validation loss
                _ = self.validate(validation_loader)
                # Get reconstruction loss for training per epoch
            self.loss["tloss_e"].append(sum(self.loss["tloss_b"][-self.total_batches:-1]) / self.total_batches)
            
            # Change learning rate if schedular=True
            _ = self.scheduler.step() if self.options["scheduler"] else None
        
        # Save plot of training and validation losses
        save_loss_plot(self.loss, self._plots_path)
        # Convert loss dictionary to a dataframe
        loss_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in self.loss.items()]))
        # Save loss dataframe as csv file for later use
        loss_df.to_csv(self._loss_path + "/losses.csv")

    def validate(self, validation_loader):
        """Computes validation loss.

        Args:
            validation_loader (): data loader for validation set.
        Returns:
            float: validation loss
        """
        with th.no_grad():
            # Compute total number of batches, assuming all test sets have same number of samples
            total_batches = len(validation_loader)
            # Initialize validation loss
            vloss = 0
            # Turn on evaluatin mode
            self.set_mode(mode="evaluation")
            # Print  validation message
            print(f"Computing validation loss. #Batches:{total_batches}")
            # Attach progress bar to data_loader to check it during training. "leave=True" gives a new line per epoch
            val_tqdm = tqdm(enumerate(validation_loader), total=total_batches, leave=True)
            # Go through batches
            for i, (x, _) in val_tqdm:

                # Generate corrupted samples -- labels are not used
                x_tilde_list, labels_list = self.subset_generator(x)
                
                # zip features and labels  -- labels are not used
                feature_label_pairs = zip(x_tilde_list, labels_list)
                # Get number of subsets, and compute their combination [((xi1, yi1), (xj1, yj1)), ((xi1, yi1), (xj2, yj2))...]
                subset_combinations = list(itertools.combinations(feature_label_pairs, 2))
                
                # Concatenate xi, and xj, and turn it into a tensor
                feature_label_batch_list = []
                for ((xi, yi), (xj, yj)) in subset_combinations:
                    Xbatch = self.process_batch(xi, xj)
                    Ybatch = self.process_batch(yi, yj)
                    feature_label_batch_list.append((Xbatch, Ybatch))    
                
                #  Concatenate original data with itself to be used when computing reconstruction error w.r.t reconstructions from xi and xj
                pretext_label = self.process_batch(x, x)
                
                val_loss = []
                # pass data through model
                for (Xtilde, Ytilde) in feature_label_batch_list:
                    # Forwards pass
                    z, latent, pretext_output = self.encoder(Xtilde)
                    # Compute losses
                    val_loss_s, _, _ = self.joint_loss(z, pretext_output, pretext_label)
                    # Accumulate losses
                    val_loss.append(val_loss_s)
                    # Delete the loss
                    del val_loss_s
                
                # Compute the validation loss for this batch
                val_loss = sum(val_loss)/len(val_loss)
                vloss = vloss + val_loss.item()
                # Clean up to avoid memory issues
                del val_loss
                gc.collect()
                    
            # Turn on training mode
            self.set_mode(mode="training")
            # Compute mean validation loss
            vloss = vloss / total_batches
            # Record the loss
            self.loss["vloss_e"].append(vloss)
            # Return mean validation loss
        return vloss

    def update_autoencoder(self, feature_label_batch_list, Xorig):
        """Updates autoencoder model"""
                
        total_loss, contrastive_loss, recon_loss = [], [], []
        
        # pass data through model
        for (Xtilde, Ytilde) in feature_label_batch_list:
            # Forwards pass
            z, latent, Xrecon = self.encoder(Xtilde)
            # If recontruct_subset is True, the output of decoder should be compared against subset (input to encoder)
            Xorig_data = Xtilde if self.options["reconstruction"] and self.options["reconstruct_subset"] else Xorig
            # Compute losses
            tloss, closs, rloss = self.joint_loss(z, Xrecon, Xorig_data)
            # Accumulate losses
            total_loss.append(tloss)
            contrastive_loss.append(closs)
            recon_loss.append(rloss)
        
        n = len(total_loss)
        total_loss = sum(total_loss)/n
        contrastive_loss = sum(contrastive_loss)/n
        recon_loss =sum(recon_loss)/n
        
        # Record reconstruction loss
        self.loss["tloss_b"].append(total_loss.item())
        self.loss["closs_b"].append(contrastive_loss.item())
        self.loss["rloss_b"].append(recon_loss.item())

        # Update Autoencoder params
        self._update_model(total_loss, self.optimizer_ae, retain_graph=True)
        # Delete loss and associated graph for efficient memory usage
        del total_loss, contrastive_loss, recon_loss, tloss, closs, rloss
        gc.collect()

    def mask_generator(self, p_m, x):
        """Generate mask vector."""
        mask = np.random.binomial(1, p_m, x.shape)
        return mask

    def subset_generator(self, x):
        """Generate corrupted samples for subsets."""
        
        n_subsets = self.options["n_subsets"]
        n_column = self.options["dims"][0]
        overlap = self.options["overlap"]
        n_column_subset = int(n_column/n_subsets)
        # Number of overlapping features between subsets
        n_overlap = int(overlap*n_column_subset)
        
        # Randomly (and column-wise) shuffle data
        column_idx = list(range(n_column))
        # Get a permutation for order of subsets
        permuted_order = np.random.permutation(n_subsets)
        # Pick subset of columns (equivalent of cropping)
        subset_column_idx_list = []
        for i in permuted_order:
            if i==0:
                start_idx = 0 if i==0 else (i+1)*n_column_subset+n_overlap
                stop_idx = (i+1)*n_column_subset+n_overlap 
            else: 
                start_idx = i*n_column_subset-n_overlap
                stop_idx = (i+1)*n_column_subset 
                
            subset_column_idx_list.append(column_idx[start_idx:stop_idx])
        
        # Get subset of features to create list of cropped data
        x_tilde_list = []
        for subset_column_idx in subset_column_idx_list:
            x_bar = x[:, subset_column_idx]
            # Add noise to cropped columns - Noise types: Zero-out, Gaussian, or Swap noise
            if self.options["add_noise"]:
                x_bar_noisy = self.generate_noisy_xbar(x_bar) 
                        
                # Generate binary mask
                p_m = self.options["masking_ratio"]
                mask = np.random.binomial(1, p_m, x_bar.shape)

                # Replace selected x_bar features with the noisy ones
                x_bar = x_bar * (1 - mask) + x_bar_noisy * mask
                
            # Add the subset to the list   
            x_tilde_list.append(x_bar)
        
        # Create arrays for labels, which are one-hot arrays showing which features are used for a particular subset
        labels_list = [np.zeros(x.shape) for i in range(n_subsets)]
        # Write 1s on clomn indexes of features used
        for i, subset_column_idx in enumerate(subset_column_idx_list): 
            labels_list[i][:, subset_column_idx] = 1.0
        
        return x_tilde_list, labels_list

    def generate_noisy_xbar(self, x):
        # Dimensions
        no, dim = x.shape
        
        # Get noise type
        noise_type = self.options["noise_type"]
        noise_level = self.options["noise_level"]
        
        # Initialize corruption array
        x_bar = np.zeros([no, dim])
        
        # Randomly (and column-wise) shuffle data
        if noise_type == "swap_noise":
            for i in range(dim):
                idx = np.random.permutation(no)
                x_bar[:, i] = x[idx, i]
        # Elif, overwrite x_bar by adding Gaussian noise to x
        elif noise_type == "gaussian_noise":
            x_bar = x + np.random.normal(0, noise_level, x.shape)
        else:
            x_bar = x_bar
        
        return x_bar
        
    def clean_up_memory(self, losses):
        """Deletes losses with attached graph, and cleans up memory"""
        for loss in losses: del loss
        gc.collect()

    def process_batch(self, xi, xj):
        """Concatenates two transformed inputs into one, and moves the data to the device as tensor"""
        # Combine xi and xj into a single batch
        Xbatch = np.concatenate((xi, xj), axis=0)
        # Convert the batch to tensor and move it to where the model is
        Xbatch = self._tensor(Xbatch)
        # Return batches
        return Xbatch

    def update_log(self, epoch, batch):
        """Updates the messages displayed during training and evaluation"""
        # For the first epoch, add losses for batches since we still don't have loss for the epoch
        if epoch < 1:
            description = f"Epoch:[{epoch - 1}], Batch:[{batch}], Total loss:{self.loss['tloss_b'][-1]:.4f}"
            description += f" contrastive loss:{self.loss['closs_b'][-1]:.4f},  recon loss:{self.loss['rloss_b'][-1]:.4f}"
        # For sub-sequent epochs, display only epoch losses.
        else:
            description = f"Epoch:[{epoch - 1}] training loss:{self.loss['tloss_e'][-1]:.4f} "
            description += f", contrastive loss:{self.loss['closs_b'][-1]:.4f},  recon loss:{self.loss['rloss_b'][-1]:.4f}"
            description += f", val loss:{self.loss['vloss_e'][-1]:.4f}"  if  self.options["validate"] else ""
        # Update the displayed message
        self.train_tqdm.set_description(description)

    def set_mode(self, mode="training"):
        """Sets the mode of the models, either as .train(), or .eval()"""
        for _, model in self.model_dict.items():
            model.train() if mode == "training" else model.eval()

    def save_weights(self):
        """Used to save weights."""
        for model_name in self.model_dict:
            th.save(self.model_dict[model_name], self._model_path + "/" + model_name + ".pt")
        print("Done with saving models.")

    def load_models(self):
        """Used to load weights saved at the end of the training."""
        for model_name in self.model_dict:
            model = th.load(self._model_path + "/" + model_name + ".pt", map_location=self.device)
            setattr(self, model_name, model.eval())
            print(f"--{model_name} is loaded")
        print("Done with loading models.")

    def print_model_summary(self):
        """Displays model architectures as a sanity check to see if the models are constructed correctly."""
        # Summary of the model
        description = f"{40 * '-'}Summary of the models (an Autoencoder and Projection network):{40 * '-'}\n"
        description += f"{34 * '='}{self.options['model_mode'].upper().replace('_', ' ')} Model{34 * '='}\n"
        description += f"{self.encoder}\n"
        # Print model architecture
        print(description)

    def _update_model(self, loss, optimizer, retain_graph=True):
        """Does backprop, and updates the model parameters

        Args:
            loss ():
            optimizer ():
            retain_graph (bool):

        Returns:
            None
        """
        # Reset optimizer
        optimizer.zero_grad()
        # Backward propagation to compute gradients
        loss.backward(retain_graph=retain_graph)
        # Update weights
        optimizer.step()

    def _set_scheduler(self):
        """Sets a scheduler for learning rate of autoencoder"""
        # Set scheduler (Its use will be optional)
        self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer_ae, step_size=1, gamma=0.99)

    def _set_paths(self):
        """ Sets paths to bse used for saving results at the end of the training"""
        # Top results directory
        self._results_path = os.path.join(self.options["paths"]["results"], self.options["framework"])
        # Directory to save model
        self._model_path = os.path.join(self._results_path, "training", self.options["model_mode"], "model")
        # Directory to save plots as png files
        self._plots_path = os.path.join(self._results_path, "training", self.options["model_mode"], "plots")
        # Directory to save losses as csv file
        self._loss_path = os.path.join(self._results_path, "training", self.options["model_mode"], "loss")

    def _adam(self, params, lr=1e-4):
        """Sets up Adam optimizer using model params"""
        return th.optim.AdamW(itertools.chain(*params), lr=lr, betas=(0.9, 0.999), eps=1e-07)

    def _tensor(self, data):
        """Turns numpy arrays to torch tensors"""
        if type(data).__module__ == np.__name__:
            data = th.from_numpy(data)
        return data.to(self.device).float()
