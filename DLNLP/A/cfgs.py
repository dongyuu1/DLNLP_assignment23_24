# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# General configs
# -----------------------------------------------------------------------------
_C = CfgNode()

_C.DATA_ROOT_PATH = "./Datasets"

_C.PT_PATH = "./A/pretrained_models/"

_C.DATA_SOURCE = "laptop"

_C.SAVE_DIR = "./"

_C.DEVICE = "cuda:0"

_C.LOG_INTERVAL = 5

_C.RAND_SEED = 10

_C.MAX_SEQ_LENGTH = 128
# -----------------------------------------------------------------------------
# Configs for training model
# -----------------------------------------------------------------------------
_C.TRAIN = CfgNode()

_C.TRAIN.BATCH = 16

_C.TRAIN.LR = 1e-5

_C.TRAIN.WARMUP_RATIO = 0.1

_C.TRAIN.EPOCH = 10

# -----------------------------------------------------------------------------
# Configs for testing model
# -----------------------------------------------------------------------------

_C.TEST = CfgNode()

_C.TEST.BATCH = 64

# -----------------------------------------------------------------------------
# Configs for model architecture
# -----------------------------------------------------------------------------

_C.MODEL = CfgNode()

_C.MODEL.HIDDEN_SIZE = 768

_C.MODEL.EPSILON = 1

_C.MODEL.LAMBDA = 10
def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
