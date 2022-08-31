import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import KFold
from torch_geometric.data import Data

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
        dataset : GraphDataset, kf_split: int, batch_size: int, logger: logging.Logger
    ) -> List[Tuple[int, FewShotDataLoader, Data]]:
        """
        Setup the kfold validation, i.e., returns a list of 
        triple (fold index, train dataloader, test data).

        :param dataset: the dataset to split
        :param kf_slip: the total number of split
        :param batch_size: the batch_size argument to the dataloader
        :param logger: a simple logger
        :return: a list of triple (fold index, train dataloader, test data)
        """
        # Create the splitter and the dataloaders list
        kfold_splitter = KFold(n_splits=kf_split, shuffle=True)
        tt_list = []

        for fold_num, (train_ids, test_ids) in enumerate(kfold_splitter.split(dataset)):
            logger.debug(f"Creating Fold number {fold_num}")

            # Take dataset subsets
            train_ds = GraphDataset.get_dataset_from_labels(dataset, train_ids)
            validation_ds = GraphDataset.get_dataset_from_labels(dataset, test_ids)

            # Get dataloaders
            train_dl = get_dataloader(
                train_ds, n_way=config.TRAIN_WAY, k_shot=config.TRAIN_SHOT,
                n_query=config.TRAIN_QUERY, epoch_size=config.TRAIN_EPISODE,
                shuffle=True, batch_size=batch_size
            )

            val_data = validation_ds.to_data()

            tt_list.append((fold_num, train_dl, val_data))
        
        return tt_list
    
    @staticmethod
    def kFold_validation(trainer: 'KFoldTrainer', logger: logging.Logger) -> None:

        @wrapt.decorator
        def wrapper(fun, *args, **kwargs) -> None:
            print("Starting kFold-Cross Validation with: K = ", config.N_FOLD)

            # Setup the KFold Cross Validation
            dataloaders = KFoldCrossValidationWrapper.setup_kFold_validation(
                dataset=trainer.dataset, kf_split=config.N_FOLD, 
                batch_size=trainer.batch_size, logger=logger
            )

            for fold, train_dl, val_data in dataloaders:
                print(f"Folder Number: {fold + 1}")
                print("---------------------------------------------------------------")
                
                # Run the wrapper function
                _ = fun(train_dl, *args, **kwargs)

                print("---------------------------------------------------------------")
                print(f"End learning with fold: {fold + 1}...")
                print(f"Start testing with fold: {fold + 1} ...")
                print("---------------------------------------------------------------")

                with torch.no_grad():
                    net = trainer.model

                    # To GPU if necessary
                    if config.DEVICE != "cpu":
                        val_data = val_data.pin_memory()
                        val_data = val_data.to(config.DEVICE)
                    
                    logits, _, _ = net(val_data.x, val_data.edge_idex, val_data.batch)
                    loss = nn.CrossEntropyLoss()(logits, val_data.y)
                    preds = F.softmax(logits, dim=1).argmax(dim=1)