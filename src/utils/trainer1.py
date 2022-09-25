import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.data as pyg_data
import numpy as np

from algorithms.asmaml.asmaml1 import AdaptiveStepMAML
from algorithms.gmixup.gmixup import OHECrossEntropy
from models.sage4maml import SAGE4MAML
from models.gcn4maml import GCN4MAML
from data.dataset import GraphDataset, OHGraphDataset
from data.dataloader1 import get_dataloader, FewShotDataLoader, GraphDataLoader
from utils.utils import elapsed_time
from typing import Tuple, List, Optional, Any
from tqdm import tqdm

import config
import os
import sys
import logging


class Trainer(object):
    """
    A simple trainer class to train a model
    on the train and validation set

    Args:
        train_ds (GraphDataset or OHGraphDataset): the train set
        validation_ds (GraphDataset or OHGraphDataset): the validation set
        logger (logging.Logger): a simple logger
        model_name (str, default=sage): the name of the model to use
        meta_model (Optional[AdaptiveStepMAML], default=None): the meta model class to use
        save_suffix (str, default=""): the suffix for saving the torch model
        dl_type (FewShotDataLoader | GraphDataLoader, default=FewShotDataLoader):
            the type of the dataloader to use for training and validation
        dataset_name (str, default="TRIANGLES"): the name of the dataset
        epochs (int, default=200): the total number of epochs
        use_mevolve (bool, default=False): True if use M-Evolve or not
        use_flag (bool, default=False): True if use FLAG or not
        use_gmixup (bool, default=False): True if use G-Mixup or not
        save_path (str, default="../models"): The folder where to save the model
        device (str, default="cpu"): The device to use
        batch_size (int, default=1): The size of a single batch
        outer_lr (float, default=0.001): The LR of the outer loop of MAML
        inner_lr (float, default=0.01): The LR of the inner loop of MAML
        stop_lr (float, default=0.0001): The LR of the stop control model
        weight_decay (float, default=1e-05): The Weight Decay of the optimizer
        max_step (int, default=15): max step
        min_step (int, default=5): min step
        penalty (float, default=0.001): the penalty for each step
        train_shot (int, default=10): the number of support graphs for each class of train
        val_shot (int, default=10): the number of support graphs for each class of validation
        train_query (int, default=15): the number of query graphs for each class of train
        val_query (int, default=15): the number of query graphs for each calss of validation
        train_way (int, default=3): the number of class to sample for train
        test_way (int, default=3): the number of class to sample for validation/testing
        val_episode (int, default=200): the number of episode for validation
        train_episode (int, default=200): the number of episode for training
        batch_episode (int, default=5): how many batch per episode
        patience (int, default=35): the patience
        grad_clip (int, default=5): grad clipping value
        scis (int, default=2): stop control model input size
        schs (int, default=20): stop control model hidden size
        beta (float, default=0.15): augmentation budget for M-Evolve
        n_fold (int, default=5): total number of fold
        n_xval (int, default=10): total number of cross-validation to run
        iters (int, default=5): total number of M-Evolve iterations
        heuristic (str, default="random_mapping"): M-Evolve heuristic to use
        lrts (int, default=1000): total number of step when computing the threshold in M-Evolve
        lrtb (int, default=30): beta value for sign function approximation
        flag_m (int, default=3): FLAG iterations number
        ass (float, default=8e-03): The attack step size of FLAG
        file_log (bool, default=False): True if log into file or not
    """
    def __init__(
        self, train_ds: GraphDataset | OHGraphDataset, val_ds: GraphDataset | OHGraphDataset,
        logger: logging.Logger, model_name: str="sage", meta_model: Optional[AdaptiveStepMAML]=None,
        save_suffix: str="_", dl_type: FewShotDataLoader | GraphDataLoader=FewShotDataLoader,
        dataset_name: str="TRIANGLES", epochs: int=200, use_mevolve: bool=False, use_flag: bool=False,
        use_gmixup: bool=False, save_path: str=config.MODELS_SAVE_PATH, device: str=config.DEVICE,
        batch_size: int=config.BATCH_SIZE, outer_lr: float=config.OUTER_LR, inner_lr: float=config.INNER_LR,
        stop_lr: float=config.STOP_LR, weight_decay: float=config.WEIGHT_DECAY, max_step: int=config.MAX_STEP,
        min_step: int=config.MIN_STEP, penalty: float=config.STEP_PENALITY, train_shot: int=config.TRAIN_SHOT,
        val_shot: int=config.VAL_SHOT, train_query: int=config.TRAIN_QUERY, val_query: int=config.VAL_QUERY,
        train_way: int=config.TRAIN_WAY, test_way: int=config.TEST_WAY, val_episode: int=config.VAL_EPISODE,
        train_episode: int=config.TRAIN_EPISODE, batch_episode: int=config.BATCH_PER_EPISODES, 
        patience: int=config.PATIENCE, grad_clip: int=config.GRAD_CLIP, scis: int=config.STOP_CONTROL_INPUT_SIZE,
        schs: int=config.STOP_CONTROL_HIDDEN_SIZE, beta: float=config.BETA, n_fold: int=config.N_FOLD, 
        n_xval: int=config.N_CROSSVALIDATION, iters: int=config.ITERATIONS, heuristic: str=config.HEURISTIC,
        lrts: int=config.LABEL_REL_THRESHOLD_STEPS, lrtb: int=config.LABEL_REL_THRESHOLD_BETA,
        flag_m: int=config.M, ass: float=config.ATTACK_STEP_SIZE, file_log: bool=False, **kwargs
    ) -> None:
        # .
        self.train_ds      = train_ds
        self.validation_ds = val_ds
        self.logger        = logger
        self.model_name    = model_name
        self.meta_model    = meta_model
        self.save_suffix   = save_suffix
        self.dl_type       = dl_type
        self.dataset_name  = dataset_name
        self.epochs        = epochs
        self.use_mevolve   = use_mevolve
        self.use_flag      = use_flag
        self.use_gmixup    = use_gmixup
        self.save_path     = save_path
        self.device        = device
        self.batch_size    = batch_size
        self.outer_lr      = outer_lr
        self.inner_lr      = inner_lr
        self.stop_lr       = stop_lr
        self.weight_decay  = weight_decay
        self.max_step      = max_step
        self.min_step      = min_step
        self.penalty       = penalty
        self.train_shot    = train_shot
        self.val_shot      = val_shot
        self.train_query   = train_query
        self.val_query     = val_query
        self.train_way     = train_way
        self.test_way      = test_way
        self.val_episode   = val_episode
        self.train_episode = train_episode
        self.batch_episode = batch_episode
        self.patience      = patience
        self.grad_clip     = grad_clip
        self.scis          = scis
        self.schs          = schs
        self.beta          = beta
        self.n_fold        = n_fold
        self.n_xval        = n_xval
        self.iters         = iters
        self.heuristic     = heuristic
        self.lrts          = lrts
        self.lrtb          = lrtb
        self.flag_m        = flag_m
        self.ass           = ass
        self.file_log      = file_log

        self.shuffle = True

        # Control if the two datasets have OHE labels
        self.is_oh_train = isinstance(self.train_ds, OHGraphDataset)
        self.is_oh_validation = isinstance(self.validation_ds, OHGraphDataset)

        # Create dataloader
        self.train_dl, self.validation_dl = self._get_dataloaders()

        # Create the base model
        self.model = self._get_model()
        self.model2save = self.model

        # Create the meta model if required
        self.meta = None
        if self.meta_model is not None:
            self.meta = self._get_meta()

        self.save_string = self._build_save_string()

    def _build_save_string(self) -> str:
        """ Return the name of the file where to save the model """
        save_str = f"{self.dataset_name}_{self.model.__class__.__name__}_"

        if self.meta_model is not None:
            save_str += f"{self.meta.__class__.__name__}_"

        if self.use_mevolve:
            return save_str + "MEvolve_bestModel.pth"

        if self.use_flag:
            return save_str + "FLAG_bestModel.pth"

        if self.use_gmixup:
            return save_str + "GMixup_bestModel.pth"

        return save_str + "bestModel.pth"

    def _get_dataloaders(self) -> Tuple[FewShotDataLoader | GraphDataLoader, FewShotDataLoader | GraphDataLoader]:
        """ Returns two dataloader: one for train and the other for validation """
        train_dataloader = get_dataloader(
            self.train_ds, self.train_way, self.train_shot, 
            self.train_query, self.train_episode, self.shuffle, 
            self.batch_size, dl_type=self.dl_type)

        self.logger.debug("Created dataloader for training of type: {}".format(
            train_dataloader.__class__.__name__
        ))

        validation_dataloader = get_dataloader(
            self.validation_ds, self.test_way, self.val_shot, 
            self.val_query, self.val_episode, self.shuffle, 
            self.batch_size, dl_type=self.dl_type)

        self.logger.debug("Created dataloader for validation of type: {}".format(
            validation_dataloader.__class__.__name__
        ))

        return train_dataloader, validation_dataloader

    def _get_model(self) -> GCN4MAML | SAGE4MAML:
        """ Return one between GCN or SAGE model """
        models = {"sage" : SAGE4MAML, "gcn" : GCN4MAML}
        model = models[self.model_name](
            num_classes=self.train_way, paper=False,
            num_features=config.NUM_FEATURES[self.dataset_name]
        ).to(self.device)

        self.logger.debug(f"Created model of type {model.__class__.__name__}")
        return model

    def _get_meta(self) -> AdaptiveStepMAML:
        """ Return the meta model """
        self.logger.debug("Creating the meta-model of type {}".format(
            self.meta_model.__name__
        ))

        configurations = {
            "inner_lr"           : self.inner_lr,
            "train_way"          : self.train_way,
            "train_shot"         : self.train_shot,
            "train_query"        : self.train_query,
            "grad_clip"          : self.grad_clip,
            "batch_per_episodes" : self.batch_episode,
            "flexible_step"      : config.FLEXIBLE_STEP,
            "min_step"           : self.min_step,
            "max_step"           : self.max_step,
            "step_test"          : config.STEP_TEST,
            "step_penalty"       : self.penalty,
            "use_score"          : config.USE_SCORE,
            "use_loss"           : config.USE_LOSS,
            "outer_lr"           : self.outer_lr,
            "stop_lr"            : self.stop_lr,
            "patience"           : self.patience,
            "weight_decay"       : self.weight_decay,
            "scis"               : self.scis,
            "schs"               : self.schs
        }

        meta = self.meta_model(self.model, False, **configurations).to(self.device)
        self.model2save = meta

        if self.is_oh_train or self.is_oh_validation:
            meta.loss = OHECrossEntropy()

        return meta

    def _train_step(self, support_data: pyg_data.Data, query_data: pyg_data.Data) -> Any:
        """ Run a single train step """
        if self.device != "cpu":
            support_data = support_data.to(self.device)
            query_data = query_data.to(self.device)

        accs, step, final_loss, total_loss, *_ = self.meta(support_data, query_data)
        return accs, step, final_loss, total_loss

    def _train(self, epoch: int=0) -> Tuple[List[float], List[float], List[float]]:
        """ Run the training """
        print("Starting Training phase", file=sys.stdout if not self.file_log else open(
            self.logger.handlers[1].baseFilename, mode="a"
        ))

        # Set training mode
        self.meta.train()

        train_accs, train_final_losses, train_total_losses = [], [], []
        for i, data in enumerate(tqdm(self.train_dl(epoch)), 1):
            support_data, _, query_data, _ = data
            accs, step, final_loss, total_loss = self._train_step(support_data, query_data)

            train_accs.append(accs[step])
            train_final_losses.append(final_loss)
            train_total_losses.append(total_loss)

            if (i + 1) % 100 == 0:
                print("({:d}) Mean Accuracy: {:.6f}, Mean Final Loss: {:.6f}, Mean Total Loss: {:.6f}".format(
                    np.mean(train_accs), np.mean(train_final_losses), np.mean(train_total_losses)
                ), file=sys.stdout if not self.file_log else open(
                        self.logger.handlers[1].baseFilename, mode="a"
                ))

        print("Ended Training Phase", file=sys.stdout if not self.file_log else open(
            self.logger.handlers[1].baseFilename, mode="a"
        ))
        return train_accs, train_final_losses, train_total_losses

    def _validation_step(self, support_data: pyg_data.Data, query_data: pyg_data.Data) -> Any:
        """ Run a single validation step """
        if self.device != "cpu":
            support_data = support_data.to(self.device)
            query_data = query_data.to(self.device)

        accs, step, _, _, query_losses = self.meta.finetuning(support_data, query_data)
        return accs, step, query_losses

    def _validation(self, epoch: int=0) -> List[float]:
        """ Run the validation phase """
        print("Starting Validation Phase", file=sys.stdout if not self.file_log else open(
            self.logger.handlers[1].baseFilename, mode="a"
        ))

        # Set validation mode
        self.meta.eval()

        val_accs = []
        for i, data in enumerate(tqdm(self.validation_dl(epoch)), 1):
            support_data, _, query_data, _ = data
            accs, step, _ = self._validation_step(support_data, query_data)
            val_accs.append(accs[step])

        print("Ended Validation Phase", file=sys.stdout if not self.file_log else open(
            self.logger.handlers[1].baseFilename, mode="a"
        ))
        return val_accs

    def _run(self) -> None:
        """ Run training and validation """
        max_val_acc = 0.0
        self.logger.debug("Starting optimization")

        for epoch in range(self.epochs):
            print("=" * 103, file=sys.stdout if not self.file_log else open(
                    self.logger.handlers[1].baseFilename, mode="a"
                )
            )

            print("Epoch Number {:04d}".format(epoch), file=sys.stdout if not self.file_log else open(
                    self.logger.handlers[1].baseFilename, mode="a"
                )
            )

            train_accs, train_final_losses, _ = self._train()
            val_accs = self._validation()

            val_acc_avg = np.mean(val_accs)
            train_acc_avg = np.mean(train_accs)
            train_loss_avg = np.mean(train_final_losses)
            val_acc_ci95 = 1.96 * np.std(np.array(val_accs)) / np.sqrt(self.val_episode)
            printable_str = ""

            if val_acc_avg > max_val_acc:
                max_val_acc = val_acc_avg
                printable_str = "Epoch(*** Best ***) {:04d}\n".format(epoch)

                torch.save({'epoch' : epoch, 'embedding' : self.model2save.state_dict()
                    }, os.path.join(self.save_path, self.save_string)
                )

            else:
                printable_str = "Epoch {:04d}\n".format(epoch)

            printable_str += (
                "\tAvg Train Loss: {:.6f}, Avg Train Accuracy: {:.6f}\n" +
                "\tAvg Validation Accuracy: {:.2f} ±{:.26f}\n" +
                "\tMeta Learning Rate: {}\n" +
                "\tBest Current Validation Accuracy: {:.2f}").format(
                    train_loss_avg, train_acc_avg,
                    val_acc_avg, val_acc_ci95, 
                    self.meta.get_meta_learning_rate(), max_val_acc
                )

            print(printable_str, file=sys.stdout if not self.file_log else open(
                    self.logger.handlers[1].baseFilename, mode="a"
                )
            )

            if self.meta_model:
                self.meta_model.adapt_meta_learning_rate(train_loss_avg)

        self.logger.debug("Optimization finished")

    @elapsed_time
    def run(self) -> None:
        """ Run the entire optimization """
        self._run()