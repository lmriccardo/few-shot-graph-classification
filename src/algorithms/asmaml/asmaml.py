""" For the source of this code check-out https://github.com/NingMa-AI/AS-MAML/blob/master/models/meta_ada.py """

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.data as gdata

from models.stopcontrol import StopControl
from algorithms.gmixup.gmixup import OHECrossEntropy
from utils.utils import compute_accuracy

import numpy as np
import math


class AdaptiveStepMAML(nn.Module):
    """ The Meta-Learner Class """
    def __init__(self, model: nn.Module, paper: bool, **kwargs) -> None:
        super(AdaptiveStepMAML, self).__init__()
        
        self.net                = model
        self.paper              = paper
        self.inner_lr           = kwargs["inner_lr"]
        self.outer_lr           = kwargs["outer_lr"]
        self.stop_lr            = kwargs["stop_lr"]
        self.weight_decay       = kwargs["weight_decay"]
        self.max_step           = kwargs["max_step"]
        self.min_step           = kwargs["min_step"]
        self.grad_clip          = kwargs["grad_clip"]
        self.flexible_step      = kwargs["flexible_step"]
        self.step_test          = kwargs["step_test"]
        self.step_penality      = kwargs["step_penalty"]
        self.batch_per_episodes = kwargs["batch_per_episodes"]

        self.task_index = 1

        self.stop_prob = 0.5
        self.stop_gate = StopControl(kwargs["scis"], kwargs["schs"])

        self.meta_optim = self.configure_optimizers()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.meta_optim, mode="min", factor=0.5, 
            patience=kwargs["patience"], verbose=True, min_lr=1e-05
        )

        self.graph_embs = []
        self.graph_labels = []
        self.index = 1
        self.is_oh_labels = False

    def configure_optimizers(self):
        """Configure Optimizers"""
        return optim.Adam([
                           {'params': self.net.parameters(),       'lr': self.outer_lr},
                           {'params': self.stop_gate.parameters(), 'lr': self.stop_lr}],
                          lr=1e-04, weight_decay=self.weight_decay
               )

    @staticmethod
    def compute_loss(logits: torch.Tensor, label: torch.Tensor, is_oh_labels: bool=False) -> float:
        if not is_oh_labels:
            return nn.CrossEntropyLoss()(logits, label.long())

        return OHECrossEntropy()(logits, label)

    @staticmethod
    def smooth(weight, p=10, eps=1e-10):
        weight_abs = weight.abs()
        less = (weight_abs < math.exp(-p)).type(torch.float)
        noless = 1.0 - less
        log_weight = less * -1 + noless * torch.log(weight_abs + eps) / p
        sign = less * math.exp(p) * weight + noless * weight.sign()

        if torch.sum(torch.isnan(log_weight)) != 0:
            raise AssertionError('stop_gate input has nan')

        return log_weight, sign

    def stop(self, step: int, loss: float, node_score: torch.Tensor):
        stop_hx = None
        if step < self.max_step:
            inputs = []

            inputs += [loss.detach()]
            score = node_score.detach()
            inputs += [score]

            inputs = torch.stack(inputs, dim=0).unsqueeze(0)
            inputs = self.smooth(inputs)[0]
            stop_gate, stop_hx = self.stop_gate(inputs, stop_hx)

            return stop_gate

        return loss.new_zeros(1, dtype=torch.float)

    def adapt_meta_learning_rate(self, loss):
        self.scheduler.step(loss)
    
    def get_meta_learning_rate(self):
        epoch_learning_rate = []
        for param_group in self.meta_optim.param_groups:
            epoch_learning_rate.append(param_group['lr'])
        return epoch_learning_rate[0]

    def forward(self, support_data: gdata.batch.Batch, query_data: gdata.batch.Batch):
        if self.paper:
            (support_nodes, support_edge_index, support_graph_indicator, support_label) = support_data
            (query_nodes, query_edge_index, query_graph_indicator, query_label) = query_data

        # It is just the number of labels to predict in the query set
        query_size = query_data.y.shape[0] if not self.paper else query_label.size()[1]

        losses_q = []  # Losses on query data
        corrects, stop_gates, train_losses, train_accs, scores = [], [], [], [], []
        
        fast_parameters = list(self.net.parameters())

        for weight in self.net.parameters():
            weight.fast = None
        
        step = 0
        self.stop_prob = 0.1 if self.stop_prob < 0.1 else self.stop_prob

        # Get adaptation step
        ada_step = min(self.max_step, self.min_step + int(1.0 / self.stop_prob))

        for k in range(0, ada_step):
            # Run the i-th task and compute the loss
            support_nodes_ = support_nodes[0] if self.paper else support_data.x
            support_edge_index_ = support_edge_index[0] if self.paper else support_data.edge_index
            support_graph_indicator_ = support_graph_indicator[0] if self.paper else support_data.batch
            support_label_ = support_label[0] if self.paper else support_data.y

            logits, score, _ = self.net(support_nodes_, support_edge_index_, support_graph_indicator_)
            loss = self.compute_loss(logits, support_label_, self.is_oh_labels)
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)

            stop_pro = self.stop(k, loss, score)
            self.stop_prob = stop_pro
            stop_gates.append(stop_pro)
            scores.append(score.item())

            with torch.no_grad():
                pred = F.softmax(logits, dim=1).argmax(dim=1)
                targets = support_label_.argmax(dim=1) if self.is_oh_labels else support_label_
                correct = torch.eq(pred, targets).sum().item()
                train_accs.append(correct / support_label_.size(0))

            step = k
            train_losses.append(loss.item())

            # Compute the gradient with respect to the loss
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []
            for index, weight in enumerate(self.net.parameters()):
                if weight.fast is not None:
                    weight.fast = weight.fast - self.inner_lr * grad[index]
                else:
                    weight.fast = weight - self.inner_lr * grad[index]
                
                fast_parameters.append(weight.fast)
            
            query_nodes_ = query_nodes[0] if self.paper else query_data.x
            query_edge_index_ = query_edge_index[0] if self.paper else query_data.edge_index
            query_graph_indicator_ = query_graph_indicator[0] if self.paper else query_data.batch
            query_label_ = query_label[0] if self.paper else query_data.y

            logits_q, _ ,_= self.net(query_nodes_, query_edge_index_, query_graph_indicator_)
            loss_q = self.compute_loss(logits_q,query_label_,self.is_oh_labels)
            losses_q.append(loss_q)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                targets = query_label_.argmax(dim=1) if self.is_oh_labels else query_label_
                correct = torch.eq(pred_q, targets).sum().item()  # convert to numpy
                corrects.append(correct)
        
        final_loss = losses_q[step]
        accs = np.array(corrects) / (query_size)
        final_acc = accs[step]
        total_loss = 0

        for step, (stop_gate, step_acc) in enumerate(zip(stop_gates[self.min_step - 1:], accs[self.min_step - 1:])):
            assert stop_gate >= 0.0 and stop_gate <= 1.0, "stop_gate error value: {:.5f}".format(stop_gate)
            log_prob = torch.log(1 - stop_gate)
            tem_loss = - log_prob * ((final_acc - step_acc - (np.exp(step) - 1) * self.step_penality))
            total_loss += tem_loss

        total_loss = (total_loss + final_acc + final_loss)
        total_loss.backward()

        if self.task_index == self.batch_per_episodes:
            if self.grad_clip > 0.1:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

            self.meta_optim.step()
            self.meta_optim.zero_grad()
            self.task_index = 1
        else:
            self.task_index += 1

        stop_gates = [stop_gate.item() for stop_gate in stop_gates]

        return accs * 100, step, final_loss.item(), total_loss.item(), stop_gates, scores, train_losses, train_accs

    def finetuning(self, support_data, query_data):

        if self.paper:
            (support_nodes, support_edge_index, support_graph_indicator, support_label) = support_data
            (query_nodes, query_edge_index, query_graph_indicator, query_label) = query_data

        # It is just the number of labels to predict in the query set
        query_size = query_data.y.shape[0] if not self.paper else query_label.size()[1]

        corrects = []
        step = 0
        stop_gates, scores, query_loss = [], [], []

        fast_parameters = list(self.net.parameters())

        for weight in self.net.parameters():
            weight.fast = None
        
        ada_step = min(self.step_test, self.min_step + int(2 / self.stop_prob))

        for k in range(ada_step):
            support_nodes_ = support_nodes[0] if self.paper else support_data.x
            support_edge_index_ = support_edge_index[0] if self.paper else support_data.edge_index
            support_graph_indicator_ = support_graph_indicator[0] if self.paper else support_data.batch
            support_label_ = support_label[0] if self.paper else support_data.y

            logits,score,_ = self.net(support_nodes_, support_edge_index_, support_graph_indicator_)
            loss = self.compute_loss(logits, support_label_, False)

            stop_pro = self.stop(k, loss, score)
            self.stop_prob = stop_pro
            stop_gates.append(stop_pro)

            step = k
            scores.append(score.item())

            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []

            for index, weight in enumerate(self.net.parameters()):
                if weight.fast is None:
                    weight.fast = weight - self.inner_lr * grad[index]
                else:
                    weight.fast = weight.fast - self.inner_lr * grad[index]

                fast_parameters.append(weight.fast)

            query_nodes_ = query_nodes[0] if self.paper else query_data.x
            query_edge_index_ = query_edge_index[0] if self.paper else query_data.edge_index
            query_graph_indicator_ = query_graph_indicator[0] if self.paper else query_data.batch
            query_label_ = query_label[0] if self.paper else query_data.y

            logits_q, _, graph_emb = self.net(query_nodes_, query_edge_index_, query_graph_indicator_)
            self.graph_labels.append(query_label_.reshape(-1))
            self.graph_embs.append(graph_emb)

            if self.index % 1 == 0:
                self.index = 1
                self.graph_embs = []
                self.graph_labels = []
            else:
                self.index += 1
            
            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, query_label_).sum().item()  # convert to numpy
                corrects.append(correct)
                loss_query = self.compute_loss(logits_q, query_label_)
                query_loss.append(loss_query.item())

        accs = 100 * np.array(corrects) / query_size
        stop_gates = [stop_gate.item() for stop_gate in stop_gates]

        return accs, step, stop_gates, scores, query_loss