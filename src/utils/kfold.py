from sklearn.model_selection import KFold
from data.dataset import GraphDataset
from data.dataloader import get_dataloader, FewShotDataLoader
from typing import List, Tuple, Optional

import config
import logging
import wrapt


class KFoldCrossValidationWrapper:
    """A simple K-Fold Cross Validator Wrapper"""

    @staticmethod
    def setup_kFold_validation(
        dataset : GraphDataset, kf_split: int, batch_size: int
    ) -> List[Tuple[int, FewShotDataLoader, FewShotDataLoader]]:
        """
        Setup the kfold validation, i.e., returns a list of 
        triple (fold index, train dataloader, test dataloader).

        :param dataset: the dataset to split
        :param kf_slip: the total number of split
        :param batch_size: the batch_size argument to the dataloader
        :return: a list of triple (fold index, train dataloader, test dataloader)
        """
        # Create the splitter and the dataloaders list
        kfold_splitter = KFold(n_splits=kf_split, shuffle=True)
        tt_list = []

        for fold_num, (train_ids, test_ids) in enumerate(kfold_splitter.split(dataset)):
            ...