import argparse
import os
import numpy as np
import random
import torch
from A import cfgs
from A import asc_launch as launch


def load_args():
    """
    Load the configuration parameters of this task
    :return: The corresponding configuration parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="A",
        help="which task to perform. possible choices: A, B",
    )

    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--epsilon", type=float, default=1, help="intensity of adversarial attack")
    parser.add_argument("--lambda_", type=float, default=10, help="weight of adversarial loss")
    args = parser.parse_args()

    return args


def load_config(args):
    """
    Construct the configuration object used for running

    :param args: The input parameters
    :return: The total configuration parameters
    """
    path = os.getcwd()
    # Setup cfg.
    assert args.task == "A" or args.task == "B"
    if args.task == "A":
        cfg = cfgs.get_cfg()
        cfg.TRAIN.LR = args.lr
        cfg.MODEL.EPSILON = args.epsilon
        cfg.MODEL.LAMBDA = args.lambda_
    else:
        cfg = None

    return cfg


def main():

    """
    The main function switching the mode
    :return: None
    """
    # Load the input arguments
    args = load_args()

    # Merge the input arguments in the configuration.
    cfg = load_config(args)

    # Fix random seed
    random.seed(cfg.RAND_SEED)
    np.random.seed(cfg.RAND_SEED)
    torch.manual_seed(cfg.RAND_SEED)
    torch.cuda.manual_seed(cfg.RAND_SEED)
    torch.cuda.manual_seed_all(cfg.RAND_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
    launch.run(cfg)



if __name__ == "__main__":
    main()