from data.dataset import GraphDataset
from data.dataloader import FewShotDataLoader, get_dataloader
from models.asmaml.gcn4maml import GCN4MAML
from models.asmaml.sage4maml import SAGE4MAML
from models.asmaml.asmaml import AdaptiveStepMAML
from utils.utils import get_max_acc, elapsed_time, setup_seed
from typing import Union, List
from torch_geometric.data import Data
from tqdm import tqdm

import logging
import config
import numpy as np
import sys
import torch
import os


class Tester:
    """Class for run tests using the best model from training"""
    def __init__(self, test_ds: GraphDataset, logger: logging.Logger, best_model_path: str,
                       dataset_name: str="TRIANGLES", model_name: str="sage", 
                       paper: bool=False) -> None:
        self.test_ds = test_ds
        self.logger = logger
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.paper = paper
        self.best_model_path = best_model_path

        self.model = self.get_model()
        self.meta_model = self.get_meta()

        # Using the pre-trained model, i.e. the best model resulted during training
        saved_models = torch.load(self.best_model_path)
        self.meta_model.load_state_dict(saved_models["embedding"])
        self.model = self.meta_model.net

    
    def get_model(self) -> Union[GCN4MAML, SAGE4MAML]:
        """Return the model to use with the MetaModel"""
        models = {'sage': SAGE4MAML, 'gcn': GCN4MAML}
        model = models[self.model_name](num_classes=config.TRAIN_WAY, paper=self.paper).to(config.DEVICE)
        self.logger.debug(f"Creating model of type {model.__class__.__name__}")
        return model

    def get_meta(self) -> AdaptiveStepMAML:
        """Return the meta model"""
        self.logger.debug(f"Creating the AS-MAML model")
        return AdaptiveStepMAML(self.model,
                                inner_lr=config.INNER_LR,
                                outer_lr=config.OUTER_LR,
                                stop_lr=config.STOP_LR,
                                weight_decay=config.WEIGHT_DECAY,
                                paper=self.paper).to(config.DEVICE)
    
    def run_one_step_test(self, support_data: Data, query_data: Data, 
                                val_accs: List[float], query_losses_list: List[float]) -> None:
        """Run one single step of testing"""
        support_data = support_data.pin_memory()
        support_data = support_data.to(config.DEVICE)

        query_data = query_data.pin_memory()
        query_data = query_data.to(config.DEVICE)

        accs, step, _, _, query_losses = self.meta_model.finetuning(support_data, query_data)

        val_accs.append(accs[step])
        query_losses_list.extend(query_losses)
    
    def get_dataloader(self) -> FewShotDataLoader:
        """Return test dataloader"""
        self.logger.debug("--- Creating the DataLoader for Testing ---")
        test_dataloader = get_dataloader(
            ds=self.test_ds, n_way=config.TEST_WAY, k_shot=config.VAL_SHOT,
            n_query=config.VAL_QUERY, epoch_size=config.VAL_EPISODE,
            shuffle=True, batch_size=1
        )

        return test_dataloader
    
    @elapsed_time
    def test(self):
        """Run testing"""
        setup_seed(1)

        test_dl = self.get_dataloader()

        print("=" * 40 + " Starting Testing " + "=" * 40)
        self.logger.debug("Starting Testing")

        val_accs = []
        query_losses_list = []
        self.meta_model.eval()

        for _, data in enumerate(tqdm(test_dl)):
            support_data, _, query_data, _ = data
            self.run_one_step_test(support_data, query_data, val_accs, query_losses_list)
        
        val_acc_avg = np.mean(val_accs)
        val_acc_ci95 = 1.96 * np.std(np.array(val_accs)) / np.sqrt(config.VAL_EPISODE)
        query_losses_avg = np.array(query_losses_list).mean()
        query_losses_min = np.array(query_losses_list).min()

        printable_string = (
            "\nTEST FINISHED --- Results\n"        +
            "\tTesting Accuracy: {:.2f} ±{:.2f}\n" + 
            "\tQuery Losses Avg: {:.6f}\n"         +
            "\tMin Query Loss: {:.6f}\n"
            ).format(
                val_acc_avg, val_acc_ci95,
                query_losses_avg, query_losses_min
            )

        print(printable_string)
        print(printable_string, file=sys.stdout if not config.FILE_LOGGING else open(self.logger.handlers[1].baseFilename, mode="a"))