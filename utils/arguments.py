"""
Description: - Collects arguments from command line, and loads configuration from the yaml files.
             - Prints a summary of all options and arguments.
"""

import os
import pprint
import torch as th
from argparse import ArgumentParser
from os.path import dirname, abspath
from utils.utils import get_runtime_and_model_config, print_config

def get_arguments():
    # Initialize parser
    parser = ArgumentParser()
    # Dataset can be provided via command line
    parser.add_argument("-d", "--dataset", type=str, default="MNIST")
    # Whether to use GPU.
    parser.add_argument("-g", "--gpu", dest='gpu', action='store_true')
    parser.add_argument("-ng", "--no_gpu", dest='gpu', action='store_false')
    parser.set_defaults(gpu=True)
    # GPU device number as in "cuda:0". Defaul is 0.
    parser.add_argument("-cuda", "--cuda_number", type=str, default='0')
    # Experiment number if MLFlow is on
    parser.add_argument("-ex", "--experiment", type=int, default=1)
    # Return parser arguments
    return parser.parse_args()

def get_config(args):
    # Get path to the root
    root_path = dirname(abspath(__file__))
    # Get path to the runtime config file
    config = os.path.join(root_path, "config", "runtime.yaml")
    # Load runtime config from config folder: ./config/ and flatten the runtime config
    config = get_runtime_and_model_config()
    # Copy dataset names to config to use later
    config["dataset"] = args.dataset
    # Define which device to use: GPU or CPU
    config["device"] = th.device('cuda:'+args.cuda_number if th.cuda.is_available() and args.gpu else 'cpu')
    # Return
    return config

def print_config_summary(config, args):
    # Summarize config on the screen as a sanity check
    print(100*"=")
    print(f"Here is the configuration being used:\n")
    print_config(config)
    print(100*"=")
    print(f"Arguments being used:\n")
    print_config(args)
    print(100*"=")
