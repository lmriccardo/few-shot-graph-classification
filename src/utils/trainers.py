import torch
import torch.optim as optim
import torch.nn.functional as F

from data.dataset import GraphDataset
from data.dataloader import FewShotDataLoader, get_dataloader
from models.gcn4maml import GCN4MAML
from models.sage4maml import SAGE4MAML
from algorithms.asmaml.asmaml import AdaptiveStepMAML
from utils.utils import elapsed_time, setup_seed
from utils.kfold import KFoldCrossValidationWrapper

from typing import Union, Tuple, List, Optional
from torch_geometric.data import Data
from tqdm import tqdm

import logging
import config
import numpy as np
import sys
import os


class BaseTrainer:
    """
    A base trainer class for training the model 
    on the train and validaton set.

    Args:
        train_ds (GraphDataset): the train set
        val_ds (GraphDataset): the validation set
        model_name (str): the name of the model to use ('sage' or 'gcn')
        logger (logging.Logger): a simple logger
        paper (bool): if paper dataset is used or not
        epochs (int, default=200): number of total epochs to run
        dataset_name (str, default=TRIANGLES): the name of the used dataset
    """
    def __init__(self, train_ds: GraphDataset, val_ds: GraphDataset,
                       logger: logging.Logger, model_name: str="sage", 
                       paper: bool=False, epochs: int=200, batch_size: int=1,
                       dataset_name: str="TRIANGLES", save_suffix: str=""):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.model_name = model_name
        self.logger = logger
        self.paper = paper
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.batch_size = batch_size

        if self.paper:
            self.train_dl, self.val_dl = None, None
        else:
            self.train_dl, self.val_dl = self.get_dataloaders()
            
        self.model = self.get_model()
        self.meta_model = None
        self.model2save = self.model
        self.save_suffix = save_suffix

    def get_model(self) -> Union[GCN4MAML, SAGE4MAML]:
        """Return the model to use with the MetaModel"""
        models = {'sage': SAGE4MAML, 'gcn': GCN4MAML}
        model = models[self.model_name](num_classes=config.TRAIN_WAY, paper=self.paper,
                                        num_features=config.NUM_FEATURES[self.dataset_name]).to(config.DEVICE)
        self.logger.debug(f"Creating model of type {model.__class__.__name__}")
        return model
    
    def get_dataloaders(self) -> Tuple[FewShotDataLoader, FewShotDataLoader]:
        """Return train and validation dataloaders"""
        self.logger.debug("--- Creating the DataLoader for Training ---")
        train_dataloader = get_dataloader(
            ds=self.train_ds, n_way=config.TRAIN_WAY, k_shot=config.TRAIN_SHOT,
            n_query=config.TRAIN_QUERY, epoch_size=config.TRAIN_EPISODE,
            shuffle=True, batch_size=self.batch_size
        )

        self.logger.debug("--- Creating the DataLoader for Validation ---")
        validation_dataloader = get_dataloader(
            ds=self.val_ds, n_way=config.TEST_WAY, k_shot=config.VAL_SHOT,
            n_query=config.VAL_QUERY, epoch_size=config.VAL_EPISODE,
            shuffle=True, batch_size=self.batch_size
        )

        return train_dataloader, validation_dataloader

    def run_one_step_train(
        self, support_data: Data, query_data: Data, train_accs: List[float],
        train_total_losses: List[float], train_final_losses: List[float], loop_counter: int
    ) -> None:
        """Run one episode, i.e. one or more tasks, of training"""
        if config.DEVICE != "cpu":
            support_data = support_data.pin_memory()
            support_data = support_data.to(config.DEVICE)

            query_data = query_data.pin_memory()
            query_data = query_data.to(config.DEVICE)

    def run_one_step_validation(self, 
        support_data: Data, query_data: Data, val_accs: List[float]
    ) -> None:
        """Run one episode, i.e. one or more tasks, of validation"""
        if config.DEVICE != "cpu":
            support_data = support_data.pin_memory()
            support_data = support_data.to(config.DEVICE)

            query_data = query_data.pin_memory()
            query_data = query_data.to(config.DEVICE)

    def train_phase(self) -> None:
        """Run the training phase"""
        ...

    def validation_phase(self) -> None:
        """Run the validation phase"""
        ...
    
    @elapsed_time
    def train(self):
        """Run the optimization (fitting)"""
        max_val_acc = 0
        print("=" * 40 + " Starting Optimization " + "=" * 40)
        self.logger.debug("Starting Optimization")

        for epoch in range(self.epochs):
            setup_seed(epoch)
            print("=" * 103, file=sys.stdout if not config.FILE_LOGGING else open(self.logger.handlers[1].baseFilename, mode="a"))
            print("=" * 103)

            self.logger.debug("Epoch Number {:04d}".format(epoch))
            print("Epoch Number {:04d}".format(epoch))

            train_accs, train_final_losses, _ = self.train_phase()
            val_accs = self.validation_phase()

            val_acc_avg = np.mean(val_accs)
            train_acc_avg = np.mean(train_accs)
            train_loss_avg = np.mean(train_final_losses)
            val_acc_ci95 = 1.96 * np.std(np.array(val_accs)) / np.sqrt(config.VAL_EPISODE)

            if val_acc_avg > max_val_acc:
                max_val_acc = val_acc_avg
                printable_string = "Epoch(***Best***) {:04d}\n".format(epoch)

                torch.save({
                        'epoch': epoch, 
                        'embedding': self.model2save.state_dict()
                    }, os.path.join(config.MODELS_SAVE_PATH, f'{self.dataset_name}_{self.save_suffix}BestModel.pth')
                )
            else :
                printable_string = "Epoch {:04d}\n".format(epoch)
            
            printable_string += "\tAvg Train Loss: {:.6f}, Avg Train Accuracy: {:.6f}\n".format(train_loss_avg, train_acc_avg) + \
                                "\tAvg Validation Accuracy: {:.2f} ±{:.26f}\n".format(val_acc_avg, val_acc_ci95) + \
                                "\tBest Current Validation Accuracy: {:.2f}".format(max_val_acc)

            print(printable_string, file=sys.stdout if not config.FILE_LOGGING else open(self.logger.handlers[1].baseFilename, mode="a"))
            if self.meta_model:
                self.meta_model.adapt_meta_learning_rate(train_loss_avg)

        self.logger.debug("Optimization Finished")
        print("Optimization Finished")


class ASMAMLTrainer(BaseTrainer):
    """Run Training with train set and validation set for the ASMAML model"""
    def __init__(self, train_ds: GraphDataset, val_ds: GraphDataset,
                       logger: logging.Logger, model_name: str="sage", 
                       paper: bool=False, epochs: int=200, batch_size: int=1,
                       dataset_name: str="TRIANGLES", save_suffix: str="ASMAML_"
    ) -> None:
        super().__init__(
            train_ds, val_ds, logger, model_name, paper, 
            epochs, batch_size, dataset_name, save_suffix
        )
        self.meta_model = self.get_meta()
        self.model2save = self.meta_model

    def get_meta(self) -> AdaptiveStepMAML:
        """Return the meta model"""
        self.logger.debug(f"Creating the AS-MAML model")
        return AdaptiveStepMAML(self.model,
                                inner_lr=config.INNER_LR,
                                outer_lr=config.OUTER_LR,
                                stop_lr=config.STOP_LR,
                                weight_decay=config.WEIGHT_DECAY,
                                paper=self.paper).to(config.DEVICE)

    def run_one_step_train(
        self, support_data: Data, query_data: Data, train_accs: List[float],
        train_total_losses: List[float], train_final_losses: List[float], loop_counter: int
    ) -> None:
        """Run one episode, i.e. one or more tasks, of training"""
        # Set support and query data to the GPU
        super().run_one_step_train(
            support_data, query_data, train_accs,
            train_total_losses, train_final_losses, loop_counter
        )

        accs, step, final_loss, total_loss, _, _, _, _ = self.meta_model(
            support_data, query_data
        )

        train_accs.append(accs[step])
        train_final_losses.append(final_loss)
        train_total_losses.append(total_loss)

        if (loop_counter + 1) % 50 == 0:
            print(f"({loop_counter + 1})" + " Mean Accuracy: {:.6f}, Mean Final Loss: {:.6f}, Mean Total Loss: {:.6f}".format(
                np.mean(train_accs), np.mean(train_final_losses), np.mean(train_total_losses)
                ), file=sys.stdout if not config.FILE_LOGGING else open(self.logger.handlers[1].baseFilename, mode="a"))
    
    def run_one_step_validation(self, support_data: Data, 
                                      query_data: Data, 
                                      val_accs: List[float]) -> None:
        """Run one episode, i.e. one or more tasks, of validation"""
        super().run_one_step_validation(support_data, query_data, val_accs)

        accs, step, _, _, _ = self.meta_model.finetuning(support_data, query_data)
        val_accs.append(accs[step])
    
    def train_phase(self) -> Tuple[List[float], List[float], List[float]]:
        self.meta_model.train()
        train_accs, train_final_losses, train_total_losses = [], [], []

        self.logger.debug("Training Phase")

        for i, data in enumerate(tqdm(self.train_dl)):
            support_data, _, query_data, _ = data
            self.run_one_step_train(
                support_data=support_data, query_data=query_data,
                train_accs=train_accs, train_total_losses=train_total_losses,
                train_final_losses=train_final_losses, loop_counter=i
            )
        
        self.logger.debug("Ended Training Phase")

        return train_accs, train_final_losses, train_total_losses

    def validation_phase(self) -> List[float]:
        val_accs = []
        self.logger.debug("Validation Phase")
        self.meta_model.eval()
        for _, data in enumerate(tqdm(self.val_dl)):
            support_data, _, query_data, _ = data
            self.run_one_step_validation(support_data=support_data, query_data=query_data, val_accs=val_accs)
        
        self.logger.debug("Ended Validation Phase")
        return val_accs
    
    def train(self):
        return super().train()


class KFoldTrainer(BaseTrainer):
    """
    Trainer class. (1) train the classifier using a
    K-Fold Cross Validation; (2) validate the model,
    i.e. finetune the classifier, using the validation set. 
    """
    def __init__(self, train_ds: GraphDataset, val_ds: GraphDataset,
                       logger: logging.Logger, meta_model: torch.nn.Module,
                       model_name: str="sage", paper: bool=False, epochs: int=200, 
                       batch_size: int=1, dataset_name: str="TRIANGLES", save_suffix: str=""
    ) -> None:
        super().__init__(
            train_ds, val_ds, logger, model_name, paper, 
            epochs, dataset_name, save_suffix, batch_size=batch_size
        )

        self.meta_model = meta_model
        self.model2save = self.meta_model
        self.save_suffix = f"{meta_model.__class__.__name__}_KFold_"
        self.model = self.meta_model.net

    def run_one_step_train(
        self, support_data: Data, query_data: Data, train_accs: List[float],
        train_total_losses: List[float], train_final_losses: List[float], loop_counter: int
    ) -> None:
        # to GPU if needed
        super().run_one_step_train(
            support_data, query_data, train_accs,
            train_total_losses, train_final_losses, loop_counter
        )

        accs, step, final_loss, total_loss, _, _, _, _ = self.meta_model(
            support_data, query_data
        )

        train_accs.append(accs[step])
        train_final_losses.append(final_loss)
        train_total_losses.append(total_loss)

        if (loop_counter + 1) % 50 == 0:
            print(f"({loop_counter + 1})" + " Mean Accuracy: {:.6f}, Mean Final Loss: {:.6f}, Mean Total Loss: {:.6f}".format(
                np.mean(train_accs), np.mean(train_final_losses), np.mean(train_total_losses)
                ), file=sys.stdout if not config.FILE_LOGGING else open(self.logger.handlers[1].baseFilename, mode="a"))

    def run_one_step_validation(self, support_data: Data, 
                                      query_data: Data, 
                                      val_accs: List[float]) -> None:
        """Run one episode, i.e. one or more tasks, of validation"""
        super().run_one_step_validation(support_data, query_data, val_accs)

        accs, step, _, _, _ = self.meta_model.finetuning(support_data, query_data)
        val_accs.append(accs[step])

    def train_phase(self) -> None:
        train_dataloader = None

        @KFoldCrossValidationWrapper.kFold_validation(trainer=self, logger=self.logger, loss=self.meta_model.loss)
        def _train_phase(train_dl: Optional[FewShotDataLoader]=None) -> None:
            self.meta_model.train()
            train_accs, train_final_losses, train_total_losses = [], [], []

            self.logger.debug("Training Phase")

            for i, data in enumerate(tqdm(train_dl)):
                support_data, _, query_data, _ = data
                self.run_one_step_train(
                    support_data=support_data, query_data=query_data,
                    train_accs=train_accs, train_total_losses=train_total_losses,
                    train_final_losses=train_final_losses, loop_counter=i
                )
            
            self.logger.debug("Ended Training Phase")

            return train_accs, train_final_losses, train_total_losses
        
        return _train_phase(train_dataloader)

    def validation_phase(self) -> List[float]:
        val_accs = []
        self.logger.debug("Validation Phase")
        self.meta_model.eval()
        for _, data in enumerate(tqdm(self.val_dl)):
            support_data, _, query_data, _ = data
            self.run_one_step_validation(support_data=support_data, query_data=query_data, val_accs=val_accs)
        
        self.logger.debug("Ended Validation Phase")
        return val_accs
    
    def train(self):
        return super().train()