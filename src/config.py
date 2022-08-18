import os
from typing import TypeVar

import torch


ROOT_PATH = os.getcwd()

TRIANGLES_DATA_URL = "https://cloud-storage.eu-central-1.linodeobjects.com/TRIANGLES.zip"

GRAPH_ATTRIBUTE  = os.path.join(ROOT_PATH, "TRIANGLES/TRIANGLES_graph_attributes.txt")
GRAPH_LABELS     = os.path.join(ROOT_PATH, "TRIANGLES/TRIANGLES_graph_labels.txt")
NODE_NATTRIBUTE  = os.path.join(ROOT_PATH, "TRIANGLES/TRIANGLES_node_attributes.txt")
GRAPH_INDICATOR  = os.path.join(ROOT_PATH, "TRIANGLES/TRIANGLES_graph_indicator.txt")
GRAPH_A          = os.path.join(ROOT_PATH, "TRIANGLES/TRIANGLES_A.txt")

T = TypeVar('T')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOAD_DATASET = True
SAVE_PICLKE  = True

TRIANGLES_NUM_FEATURES = 1
NHID = 128
POOLING_RATIO = 0.5
DROPOUT_RATIO = 0.3

OUTER_LR     = 0.001
INNER_LR     = 0.01
STOP_LR      = 0.0001
WEIGHT_DECAY = 1E-05

MAX_STEP      = 15
MIN_STEP      = 5
STEP_TEST     = 15
FLEXIBLE_STEP = True
STEP_PENALITY = 0.001
USE_SCORE     = True
USE_GRAD      = False
USE_LOSS      = True

# Episodes: How many tasks to run

TRAIN_SHOT         = 10   # K-shot for training set
VAL_SHOT           = 10   # K-shot for validation (or test) set
TRAIN_QUERY        = 15   # Number of query for the training set
VAL_QUERY          = 15   # Number of query for the validation (or test) set
TRAIN_WAY          = 3    # N-way for training set
TEST_WAY           = 3    # N-way for test set
VAL_EPISODE        = 200  # Number of episodes for validation
TRAIN_EPISODE      = 200  # Number of episodes for training
BATCH_PER_EPISODES = 5    # How many batch per episode
EPOCHS             = 500  # How many epochs
PATIENCE           = 35
GRAD_CLIP          = 5

# Stop Control configurations
STOP_CONTROL_INPUT_SIZE = 2
STOP_CONTROL_HIDDEN_SIZE = 20