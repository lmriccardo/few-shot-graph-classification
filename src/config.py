from typing import TypeVar

import os


TRIANGLES_ZIP_URL = "https://cloud-storage.eu-central-1.linodeobjects.com/TRIANGLES.zip"
COIL_DEL_ZIP_URL = "https://cloud-storage.eu-central-1.linodeobjects.com/COIL-DEL.zip"
R52_ZIP_URL = "https://cloud-storage.eu-central-1.linodeobjects.com/R52.zip"
LETTER_HIGH_ZIP_URL = "https://cloud-storage.eu-central-1.linodeobjects.com/Letter-High.zip"

DATASETS = {
    "TRIANGLES"   : TRIANGLES_ZIP_URL, 
    "COIL-DEL"    : COIL_DEL_ZIP_URL, 
    "R52"         : R52_ZIP_URL, 
    "Letter-High" : LETTER_HIGH_ZIP_URL
}

DEFAULT_DATASET = "COIL-DEL"

T = TypeVar('T')

DEVICE = "cpu"
DOWNLOAD_DATASET = False
SAVE_PICKLE  = True
EDGELIMIT_PRINT = 2000
SAVE_PRETRAINED = True
FILE_LOGGING = False
LOGGING_PATH = os.path.abspath("../log") if FILE_LOGGING else None
DATA_PATH = os.path.abspath("../data") if not DOWNLOAD_DATASET else None
MODELS_SAVE_PATH = "../models"

NUM_FEATURES = {"TRIANGLES": 1, "R52": 1, "Letter-High": 2, "COIL-DEL": 2}

MODEL_NAME = "sage"

########################################################################################
############################### AS-MAML CONFIGURATION ##################################
########################################################################################

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

STOP_CONTROL_INPUT_SIZE = 2
STOP_CONTROL_HIDDEN_SIZE = 20


########################################################################################
############################### ML-EVOLVE CONFIGURATION ################################
########################################################################################

BETA                = 0.15
N_FOLD              = 5     # For nCross Fold Validation
N_CROSSVALIDATION   = 10    # Number of k-cross validation to run
ITERATIONS          = 5
HEURISTIC           = "random_mapping"

LABEL_REL_THRESHOLD_STEPS = 1000
LABEL_REL_THRESHOLD_BETA  = 30
LABEL_REL_THRESHOLD_STEP_SIZE = 1E-02