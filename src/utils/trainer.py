from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.data as pyg_data

from torch.nn.modules.loss import _Loss, _WeightedLoss
from data.dataset import GraphDataset, OHGraphDataset
from data.dataloader import FewShotDataLoader, GraphDataLoader, get_dataloader
from models.gcn4maml import GCN4MAML
from models.sage4maml import SAGE4MAML
from algorithms.asmaml.asmaml import AdaptiveStepMAML
from algorithms.flag.flag import FlagGDA
from algorithms.gmixup.gmixup import OHECrossEntropy
from algorithms.mevolve.mevolve import MEvolveGDA
from utils.utils import elapsed_time, setup_seed, \
                        data_batch_collate, rename_edge_indexes, \
                        compute_accuracy
from utils.kfold import KFoldCrossValidationWrapper

from typing import Union, Tuple, List, Optional, Dict, Any
from tqdm import tqdm

import logging
import config
import numpy as np
import sys
import os


class Trainer:
    """
    A simple trainer class to train a model
    on the train and validation set

    Args:
        train_ds (GraphDataset or OHGraphDataset): the train set
        val_ds (GraphDataset or OHGraphDataset): the validation set
        model_name (str): the name of the model to use ('sage' or 'gcn')
        logger (logging.Logger): a simple logger
        paper (bool): if paper dataset is used or not
        meta_model (Optional[AdaptiveStepMAML], default=None): the meta model class to use
        epochs (int, default=200): number of total epochs to run
        dataset_name (str, default=TRIANGLES): the name of the used dataset
        dataloader_type (FewShotDataLoader | GraphDataLoader, default=FewShotDataLoader):
            the type of the dataloader to use for training and validation
        use_mevolve (bool, False): True if MEvolve should be used, false otherwise
        use_flag (bool, False): True if FLAG is used, false otherwise
        use_gmixup (bool, False): True if G-Mixup is used, false otherwise
    """
    def __init__(
        self, train_ds: GraphDataset | OHGraphDataset, val_ds: GraphDataset | OHGraphDataset,
        logger: logging.Logger, meta_model: Optional[AdaptiveStepMAML]=None, save_suffix: str="_",
        dataloader_type: FewShotDataLoader | GraphDataLoader=FewShotDataLoader, paper: bool=False, **kwargs
    ) -> None:
        # .
        self.train_ds       = train_ds
        self.validation_ds  = val_ds
        self.logger         = logger
        self.paper          = paper
        self.meta_model_cls = meta_model
        self.save_suffix    = save_suffix
        self.dataloader     = dataloader_type
        self.kwargs         = kwargs
        
        self._setattr(kwargs)

        # Control if the two datasets have OHE labels
        self.is_train_oh      = isinstance(self.train_ds, OHGraphDataset)
        self.is_validation_oh = isinstance(self.validation_ds, OHGraphDataset)

        # Create dataloaders
        self.train_dl, self.validation_dl = None, None
        if not self.paper:
            self.train_dl, self.validation_dl = self._get_dataloaders()
        
        # Create the base model
        self.model = self._get_model()
        self.model2save = self.model

        # Create the meta model if necessary
        self.meta_model = None
        if self.meta_model_cls is not None:
            self.meta_model = self._get_meta_model()

    def _setattr(self, kwargs: Dict[str, Any]) -> None:
        """Set the attribute in the kwargs dict"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _get_model(self) -> Union[GCN4MAML, SAGE4MAML]:
        """Return the model to use with the MetaModel"""
        models = {"sage" : SAGE4MAML, "gcn" : GCN4MAML}
        model = models[self.model_name](
            num_classes=config.TRAIN_WAY, paper=self.paper,
            num_features=config.NUM_FEATURES[self.data_name]
        ).to(self.device)

        self.logger.debug(f"Creating model of type {model.__class__.__name__}")
        return model

    def _get_dataloaders(self) -> Tuple[
        FewShotDataLoader | GraphDataLoader, FewShotDataLoader | GraphDataLoader
    ]:
        """Return train and validation dataloader"""
        train_dataloader = get_dataloader(
            ds=self.train_ds, n_way=self.train_way, k_shot=self.train_shot,
            n_query=self.train_query, epoch_size=self.train_episode,
            shuffle=True, batch_size=self.batch_size,
            oh_labels=self.is_train_oh, dl_type=self.dataloader
        )

        self.logger.debug("Created dataloader for training of type: {}".format(
            train_dataloader.__class__.__name__
        ))

        validation_dataloader = get_dataloader(
            ds=self.validation_ds, n_way=self.test_way, k_shot=self.val_shot,
            n_query=self.val_query, epoch_size=self.val_episode,
            shuffle=True, batch_size=self.batch_size,
            oh_labels=self.is_validation_oh, dl_type=self.dataloader
        )

        self.logger.debug("Created dataloader for validation of type: {}".format(
            validation_dataloader.__class__.__name__
        ))

        return train_dataloader, validation_dataloader

    def _get_meta_model(self) -> AdaptiveStepMAML:
        """Return the meta model (in this case the only available is AS-MAML)"""
        self.logger.debug("Creating the meta-model of type {}".format(
            self.meta_model_cls.__name__
        ))

        mm_configuration = {
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

        meta = self.meta_model_cls(self.model, self.paper, **mm_configuration).to(self.device)
        self.model2save = meta
        
        if self.is_train_oh or self.is_validation_oh:
            meta.loss = OHECrossEntropy()

        return meta
    
    def _meta_train_step(
        self, support_data: pyg_data.Data, query_data: pyg_data.Data, train_accs: List[float],
        train_total_losses: List[float], train_final_losses: List[float], loop_counter: int,
        support_data_list: List[pyg_data.Data], query_data_list: List[pyg_data.Data]
    ) -> None:
        """Run one episode, i.e. one or more tasks, of training"""
        flag_data = None
        if support_data_list is not None and query_data_list is not None:
            flag_data, _ = data_batch_collate(
                rename_edge_indexes(support_data_list + query_data_list),
                oh_labels=self.is_train_oh
            )

        # If both lists are None then we cannot use FLAG
        if not flag_data:
            self.use_flag = False

        if self.device != "cpu":
            support_data = support_data.pin_memory()
            support_data = support_data.to(self.device)

            query_data = query_data.pin_memory()
            query_data = query_data.to(self.device)

        @FlagGDA.flag(
            gnn=self.model, criterion=self.meta_model.loss, data=flag_data, 
            targets=flag_data.y if flag_data is not None else None, 
            iterations=self.flag_m, step_size=self.ass, 
            use=self.use_flag, optimizer=self.meta_model.meta_optim, 
            oh_labels=self.is_train_oh, device=self.device
        )
        def _run(*args, **kwargs) -> Tuple[List[float], List[float]]:
            # If we use the GPU then we need to set the GPU
            accs, step, final_loss, total_loss, _, _, _, _ = self.meta_model(
                support_data, query_data
            )

            train_accs.append(accs[step])
            train_final_losses.append(final_loss)
            train_total_losses.append(total_loss)

            if (loop_counter + 1) % 50 == 0:
                print(f"({loop_counter + 1})" + (" Mean Accuracy: {:.6f}, Mean Final Loss: {:.6f}" +
                                                 ", Mean Total Loss: {:.6f}").format(
                        np.mean(train_accs), 
                        np.mean(train_final_losses), 
                        np.mean(train_total_losses)
                    ), file=sys.stdout if not self.file_log else open(
                            self.logger.handlers[1].baseFilename, mode="a"
                        )
                    )

            return train_accs, train_final_losses
        
        return _run()

    def _train_step(
        self, data: pyg_data.Data, train_accs: List[float],
        train_total_losses: List[float], train_final_losses: List[float], 
        loop_counter: int, criterion: _Loss | _WeightedLoss, optimizer: optim.Optimizer
    ) -> None:
        """Run one step of single and simple training"""
        if self.device != "cpu":
            data = data.to(self.device)
        
        @FlagGDA.flag(
            gnn=self.model, criterion=criterion, data=data, 
            targets=data.y, iterations=self.flag_m, step_size=self.ass, 
            use=self.use_flag, optimizer=optimizer, 
            oh_labels=self.is_train_oh, device=self.device
        )
        def _run(*args, **kwargs) -> None:
            optimizer.zero_grad()
            logits, _, _ = self.model(data.x, data.edge_index, data.batch)
            loss = criterion(logits, data.y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = F.softmax(logits, dim=1).argmax(dim=1)
                train_accs.append(compute_accuracy(preds, data.y, self.is_train_oh))
            
            train_total_losses.append(loss)
            train_final_losses.append(loss)

            if (loop_counter + 1) % 50 == 0:
                print(f"({loop_counter + 1})" + (" Mean Accuracy: {:.6f}, Mean Final Loss: {:.6f}" +
                                                 ", Mean Total Loss: {:.6f}").format(
                        np.mean(train_accs), 
                        train_final_losses[-1], 
                        np.mean(train_total_losses)
                    ), file=sys.stdout if not self.file_log else open(
                            self.logger.handlers[1].baseFilename, mode="a"
                        )
                    )
            
            return train_accs, train_final_losses
        
        return _run()

    def _meta_train(self) -> Tuple[List[float], List[float], List[float]]:
        """Run MetaTraining"""
        train_dataloader = None

        @KFoldCrossValidationWrapper.kFold_validation(
            trainer=self, logger=self.logger,
            loss=self.meta_model.loss, use=self.use_mevolve,
            oh_labels=self.is_train_oh, dl_type=self.dataloader,
            train_way=self.train_way, train_shot=self.train_shot,
            train_query=self.train_query, train_episode=self.train_episode,
            n_fold=self.n_fold, device=self.device
        )
        def _run(train_dl: Optional[FewShotDataLoader | GraphDataLoader]=None) -> None:
            self.meta_model.train()
            train_accs, train_final_losses, train_total_losses = [], [], []

            self.logger.debug("Training Phase")

            for i, data in enumerate(tqdm(train_dl)):
                support_data_list, query_data_list = None, None

                if not self.paper:
                    support_data, support_data_list, query_data, query_data_list = data
                else:
                    support_data, query_data, _ = data
                
                self._meta_train_step(
                    support_data=support_data, query_data=query_data,
                    train_accs=train_accs, train_total_losses=train_total_losses,
                    train_final_losses=train_final_losses, loop_counter=i,
                    support_data_list=support_data_list, query_data_list=query_data_list
                )
            
            self.logger.debug("Ended Training Phase")

            return train_accs, train_final_losses, train_total_losses
        
        return _run(train_dataloader)

    def _base_train(self) -> Tuple[List[float], List[float], List[float]]:
        """Run the basic training"""
        train_dataloader = None
        criterion = nn.CrossEntropyLoss() if not self.is_train_oh else OHECrossEntropy()
        optimizer = optim.Adam(self.model.parameters(), lr=self.outer_lr, weight_decay=config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=self.patience, verbose=True, min_lr=1e-5
        )

        @KFoldCrossValidationWrapper.kFold_validation(
            trainer=self, logger=self.logger,
            loss=self.meta_model.loss, use=self.use_mevolve,
            oh_labels=self.is_train_oh, dl_type=self.dataloader,
            train_way=self.train_way, train_shot=self.train_shot,
            train_query=self.train_query, train_episode=self.train_episode,
            n_fold=self.n_fold, device=self.device
        )
        def _run(train_dl: Optional[FewShotDataLoader | GraphDataLoader]=None) -> None:
            self.meta_model.train()
            train_accs, train_final_losses, train_total_losses = [], [], []

            self.logger.debug("Training Phase")

            for i, data in enumerate(tqdm(train_dl)):
                data_train, _ = data
                self._train_step(
                    data=data_train, train_accs=train_accs, train_final_losses=train_final_losses,
                    train_total_losses=train_total_losses, loop_counter=i,
                    criterion=criterion, optimizer=optimizer
                )

            scheduler.step()
            
            self.logger.debug("Ended Training Phase")

            return train_accs, train_final_losses, train_total_losses

        return _run(train_dataloader)

    def _train(self) -> Tuple[List[float], List[float], List[float]]:
        """Run the training steps"""
        if self.meta_model is not None:
            return self._meta_train()
        
        return self._base_train()

    def _val_step(self, support_data : pyg_data.Data,
                        val_accs     : List[float],
                        query_data   : Optional[pyg_data.Data]=None
    ) -> None:
        """Run validation step"""
        if self.device != "cpu":
            support_data = support_data.pin_memory()
            support_data = support_data.to(self.device)

        # If both support_data and query_data are given
        # then we have to run the finetuning of the meta model
        # otherwise, if only support_data is not None then
        # this means that we have to run a simple validation step
        if query_data is not None:
            if self.device != "cpu":
                query_data = query_data.pin_memory()
                query_data = query_data.to(self.device)

            accs, step, _, _, _ = self.meta_model.finetuning(support_data, query_data)
            val_accs.append(accs[step])

            return None
    
        logits, _, _ = self.model(support_data.x, support_data.edge_index, support_data.batch)
        preds = F.softmax(logits, dim=1).argmax(dim=1)
        val_accs.append(compute_accuracy(preds, support_data.y, self.is_validation_oh))

        return None

    def _validation(self) -> List[float]:
        """Run the validation phase"""
        val_accs = []
        self.logger.debug("Validation Phase")
        model2use = self.meta_model if self.meta_model is not None else self.model
        model2use.eval()

        for _, data in enumerate(tqdm(self.validation_dl)):
            if self.meta_model is not None:
                if not self.paper:
                    support_data, _, query_data, _ = data
                else:
                    support_data, query_data, _ = data
                    
                self._val_step(support_data, val_accs, query_data)
                continue
        
            val_data, _ = data
            self._val_step(val_data, val_accs)
        
        self.logger.debug("Ended Validation Phase")
        return val_accs
    
    def _train_run(self) -> None:
        """Run the entire training"""
        max_val_acc = 0.0
        print("=" * 40 + " Starting Optimization " + "=" * 40)
        self.logger.debug("Starting Optimization")

        for epoch in range(self.epochs):
            print("=" * 103)
            print("=" * 103, file=sys.stdout if not self.file_log else open(
                    self.logger.handlers[1].baseFilename, mode="a"
                )
            )

            train_dataloader = deepcopy(self.train_dl)
            self.train_dl = self.train_dl(epoch)

            validation_dataloader = deepcopy(self.validation_dl)
            self.validation_dl = self.validation_dl(epoch)

            self.logger.debug("Epoch Number {:04d}".format(epoch))
            print("Epoch Number {:04d}".format(epoch))

            train_accs, train_final_losses, _ = self._train()
            val_accs = self._validation()

            self.validation_dl = validation_dataloader
            self.train_dl = train_dataloader

            val_acc_avg = np.mean(val_accs)
            train_acc_avg = np.mean(train_accs)
            train_loss_avg = np.mean(train_final_losses)
            val_acc_ci95 = 1.96 * np.std(np.array(val_accs)) / np.sqrt(self.val_episode)

            if val_acc_avg > max_val_acc:
                max_val_acc = val_acc_avg
                printable_string = "Epoch(***Best***) {:04d}\n".format(epoch)

                torch.save({'epoch': epoch, 'embedding': self.model2save.state_dict()
                    }, os.path.join(
                        self.save_path, 
                        f'{self.data_name}_{self.save_suffix}BestModel.pth'
                    )
                )
            else :
                printable_string = "Epoch {:04d}\n".format(epoch)
            
            printable_string += (
                "\tAvg Train Loss: {:.6f}, Avg Train Accuracy: {:.6f}\n" +
                "\tAvg Validation Accuracy: {:.2f} ±{:.26f}\n" +
                "\tBest Current Validation Accuracy: {:.2f}").format(
                    train_loss_avg, train_acc_avg,
                    val_acc_avg, val_acc_ci95, max_val_acc
                )

            print(printable_string, file=sys.stdout if not self.file_log else open(
                    self.logger.handlers[1].baseFilename, mode="a"
                )
            )

            if self.meta_model:
                self.meta_model.adapt_meta_learning_rate(train_loss_avg)

        self.logger.debug("Optimization Finished")
        print("Optimization Finished")
    
    @elapsed_time
    def run(self) -> None:
        """Run the entire optimization (train + validation)"""
        # If use_mevolve is set to True then we need to invoke the MEvolve class
        if self.use_mevolve:
            me = MEvolveGDA(
                trainer=self, n_iters=self.iters,
                logger=self.logger, train_ds=self.train_ds,
                validation_ds=self.validation_ds,
                threshold_beta=self.lrtb, threshold_steps=self.lrts,
                heuristic=self.heuristic
            )

            _ = me.evolve()

            return None
        
        if self.use_gmixup:
            # TODO: Implement also G-Mixup optimization
            # Idea: implement a __new__ method for the trainer
            # given as input the trainer itself, but using
            # one-hot encoded labels.
            return None
        
        self._train_run()
        return None