"""
Description: Library of models and related support functions.
"""

import os
import copy
import numpy as np
import pandas as pd
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class AEWrapper(nn.Module):
    """
    Autoencoder
    """
    def __init__(self, options):
        super(AEWrapper, self).__init__()
        self.options = options
        self.encoder = Encoder(options)
        self.decoder = Decoder(options)
        
        output_dim = self.options["dims"][-1]
        # Two-Layer Projection Network
        # First linear layer, which will be followed with non-linear activation function in the forward()
        self.linear_layer1 = nn.Linear(output_dim, output_dim)
        # Last linear layer for final projection
        self.linear_layer2 = nn.Linear(output_dim, output_dim)
        
    def forward(self, x):
        # Forward pass on Encoder
        latent = self.encoder(x)
        # Apply linear layer followed by non-linear activation to decouple final output, z, from representation layer h.
        z = F.leaky_relu(self.linear_layer1(latent))
        # Apply final linear layer
        z = self.linear_layer2(z)
        # Do L2 normalization
        z = F.normalize(z, p=self.options["p_norm"], dim=1) if self.options["normalize"] else z

        # Forward pass on decoder
        x_pretext = self.decoder(latent) 
        # Return 
        return z, latent, x_pretext

    
class Encoder(nn.Module):
    """
    :param dict options: Generic dictionary to configure the model for training.
    :return: (mean, logvar) if in VAE mode. Else it return (z, z).

    Encoder model.
    """

    def __init__(self, options):
        super(Encoder, self).__init__()
        # Deepcopy options to avoid overwriting the original
        self.options = copy.deepcopy(options)  
        # Compute the shrunk size of input dimension
        n_column_subset = int(self.options["dims"][0]/self.options["n_subsets"])
        # Ratio of overlapping features between subsets
        overlap = self.options["overlap"]
        # Number of overlapping features between subsets
        n_overlap = int(overlap*n_column_subset)
        # Overwrie the input dimension
        self.options["dims"][0] = n_column_subset + n_overlap
        # Forward pass on hidden layers
        self.hidden_layers = HiddenLayers(self.options, network="encoder")
        # Compute the mean i.e. bottleneck in Autoencoder
        self.mean = nn.Linear(self.options["dims"][-2], self.options["dims"][-1])

    def forward(self, h):
        # Forward pass on hidden layers
        h = self.hidden_layers(h)
        # Compute the mean i.e. bottleneck in Autoencoder
        mean = self.mean(h)
        return mean


class Decoder(nn.Module):
    def __init__(self, options):
        super(Decoder, self).__init__()
        # Deepcopy options to avoid overwriting the original
        self.options = copy.deepcopy(options)
        # If recontruct_subset is True, output dimension is same as input dimension of Encoder. Otherwise, 
        # output dimension is same as original feature dimension of tabular data
        if self.options["reconstruction"] and self.options["reconstruct_subset"]:
            # Compute the shrunk size of input dimension
            n_column_subset = int(self.options["dims"][0]/self.options["n_subsets"])
            # Overwrie the input dimension
            self.options["dims"][0] = n_column_subset
        # Revert the order of hidden units so that we can build a Decoder, which is the symmetric of Encoder
        self.options["dims"] = self.options["dims"][::-1]
        # Add number-of-domains to the input dimension of Decoder to make it conditional
        self.options["dims"][0] = self.options["dims"][0] #+ self.options["n_domains"]
        # Add hidden layers
        self.hidden_layers = HiddenLayers(self.options, network="decoder")
        # Compute logits and probabilities
        self.logits = nn.Linear(self.options["dims"][-2], self.options["dims"][-1])

    def forward(self, h):
        # Forward pass on hidden layers
        h = self.hidden_layers(h)
        # Compute logits
        logits = self.logits(h)
        return logits

class HiddenLayers(nn.Module):
    def __init__(self, options, network="encoder"):
        super(HiddenLayers, self).__init__()
        self.layers = nn.ModuleList()
        dims = options["dims"]

        for i in range(1, len(dims) - 1):
            self.layers.append(nn.Linear(dims[i - 1], dims[i]))
            if options["isBatchNorm"]:
                self.layers.append(nn.BatchNorm1d(dims[i]))

            self.layers.append(nn.LeakyReLU(inplace=False))
            if options["isDropout"]:
                self.layers.append(nn.Dropout(options["dropout_rate"]))

    def forward(self, x):
        for layer in self.layers:
            # You could do an if isinstance(layer, nn.Type1) maybe to check for types
            x = layer(x)

        return x
