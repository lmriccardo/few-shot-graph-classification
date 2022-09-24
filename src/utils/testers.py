import torch_geometric.data as pyg_data

from data.dataset import GraphDataset, OHGraphDataset
from data.dataloader import FewShotDataLoader, GraphDataLoader, get_dataloader
from models.gcn4maml import GCN4MAML
from models.sage4maml import SAGE4MAML
from algorithms.asmaml.asmaml import AdaptiveStepMAML
from utils.utils import elapsed_time
from typing import Union, List
from tqdm import tqdm

import logging
import config
import numpy as np
import sys
import torch


class Tester(object):
    """
    Tester class

    Args:
        test_ds (GraphDataset | OHGraphDataset): the dataset for testing
        model_filename (str): the path of the saved torch model
        logger (logging.Logger): a sinple logger
        model (Optional[AdaptiveStepMAML], default=None): the model to test
        test_way (int, default=3): number of classes for test
        test_shot (int, default=10): number of support graphs per class
        test_query (int, default=15): number of query graphs per class
        batch_size (int, default=1): the size of each batch
        test_episode (int, default=200): number of batch per episode
        device (str, default="cpu"): the device to use
        dl_type (FewShotDataLoader | GraphDataLoader, default=FewShotDataLoader):
            the type of dataloader to use
    """
    def __init__(
        self, test_ds: GraphDataset | OHGraphDataset, model_filename: str, logger: logging.Logger,
        model: AdaptiveStepMAML, test_way: int=config.TEST_WAY, test_shot: int=config.VAL_SHOT, 
        test_query: int=config.VAL_QUERY, batch_size: int=config.BATCH_SIZE, test_episode: int=config.VAL_EPISODE,
        device: str=config.DEVICE, dl_type: FewShotDataLoader | GraphDataLoader=FewShotDataLoader
    ) -> None:
        #.
        self.test_ds        = test_ds
        self.model_filename = model_filename
        self.model          = model
        self.test_way       = test_way
        self.test_shot      = test_shot
        self.test_query     = test_query
        self.batch_size     = batch_size
        self.test_episode   = test_episode
        self.device         = device
        self.dl_type        = dl_type

        self.shuffle = True

        # Load the saved model
        self._load()

        self.test_dl = self._get_dataloader()

    def _load(self) -> None:
        """ Load the saved state dict into the model """
        state_dict = torch.load(self.model_filename)["embedding"]
        self.model.load_state_dict(state_dict)

    def _get_dataloader(self) -> FewShotDataLoader | GraphDataLoader:
        """ Return a dataloader for the test set """
        test_dataloader = get_dataloader(
            self.test_ds, self.test_way, self.test_shot, 
            self.test_query, self.test_episode, 
            self.shuffle, self.batch_size, dl_type=self.dl_type)

        self.logger.debug("Created dataloader for testing of type: {}".format(
            test_dataloader.__class__.__name__
        ))

        return test_dataloader

    def _test_step(self, support_data: pyg_data.Data, query_data: pyg_data.Data) -> Any:
        """ Run a single step of test """
        if self.device != "cpu":
            support_data = support_data.to(self.device)
            query_data = query_data.to(self.device)

        accs, step, _, _, query_losses = self.model.finetuning(support_data, query_data)
        return accs, step, query_losses

    def _test(self) -> List[float]:
        """ Run the test """
        val_accs = []
        self.model.eval()
        for _, data in enumerate(tqdm(self.test_dl(1)), 1):
            support_data, _, query_data, _ = data
            accs, step, _ = self._test_step(support_data, query_data)
            val_accs.append(accs[step])

        return val_accs
    
    @elapsed_time
    def test(self) -> None:
        """ Run test """
        self.logger.debug("Start Testing")

        val_accs = self._test()
        val_acc_avg = np.mean(val_accs)
        val_acc_ci95 = 1.96 * np.std(np.array(val_accs)) / np.sqrt(self.test_episode)

        print("Final Resulting Accuracy: {:.2f} ±{:.26f}".format(val_acc_avg, val_acc_ci95))
        self.logger.debug("Ended Testing")