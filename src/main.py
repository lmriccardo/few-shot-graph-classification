import sys
import os
sys.path.append(os.getcwd())

from data.dataset import generate_train_val_test
from utils.utils import delete_data_folder


train_ds, test_ds, val_ds = generate_train_val_test(download_data=True, perc_train=50, perc_test=30)
print(train_ds, test_ds, val_ds)

delete_data_folder()