""" For the source of this code check-out https://github.com/NingMa-AI/AS-MAML/blob/master/models/meta_ada.py """

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch_geometric.data as gdata

import numpy as np

from models.asmaml.stopcontrol import StopControl

import config
import math


class AdaptiveStepMAML(nn.Module):
    """ The Meta-Learner Class """
    def __init__(self, model, inner_lr, outer_lr, stop_lr, weight_decay, paper) -> None:
        super().__init__()
        self.net          = model
        self.inner_lr     = inner_lr
        self.outer_lr     = outer_lr
        self.stop_lr      = stop_lr
        self.weight_decay = weight_decay
        self.paper        = paper

        self.task_index = 1

        self.stop_prob = 0.5
        self.stop_gate = StopControl(config.STOP_CONTROL_INPUT_SIZE, config.STOP_CONTROL_HIDDEN_SIZE)

        self.meta_optim = self.configure_optimizers()

        self.loss      = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.meta_optim, mode="min", factor=0.5, 
            patience=config.PATIENCE, verbose=True, min_lr=1e-05
        )

        self.graph_embs = []
        self.graph_labels = []
        self.index = 1

    def configure_optimizers(self):
        """Configure Optimizers"""
        return optim.Adam([
                           {'params': self.net.parameters(),       'lr': self.outer_lr},
                           {'params': self.stop_gate.parameters(), 'lr': self.stop_lr}],
                          lr=1e-04, weight_decay=self.weight_decay
               )
        
    def compute_loss(self, logits: torch.Tensor, label: torch.Tensor) -> float:
        return self.loss(logits, label.long())

    @staticmethod
    def smooth(weight, p=10, eps=1e-10):
        weight_abs = weight.abs()
        less = (weight_abs < math.exp(-p)).type(torch.float)
        noless = 1.0 - less
        log_weight = less * -1 + noless * torch.log(weight_abs + eps) / p
        sign = less * math.exp(p) * weight + noless * weight.sign()
        assert  torch.sum(torch.isnan(log_weight))==0,'stop_gate input has nan'
        return log_weight, sign

    def stop(self, step: int, loss: float, node_score: torch.Tensor):
        stop_hx = None
        if config.FLEXIBLE_STEP and step < config.MAX_STEP:
            inputs = []

            if config.USE_LOSS:
                inputs += [loss.detach()]
            if config.USE_SCORE:
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
        # In the paper's code, they compute the so-called task_num. 
        # However, it's value is always 1. For this reason, I decided to not use it
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
        ada_step = min(config.MAX_STEP, config.MIN_STEP + int(1.0 / self.stop_prob))

        for k in range(0, ada_step):
            # Run the i-th task and compute the loss
            if not self.paper:
                logits, score, _ = self.net(support_data.x, support_data.edge_index, support_data.batch)
                loss = self.compute_loss(logits, support_data.y)
            else:
                logits, score, _ = self.net(support_nodes[0], support_edge_index[0], support_graph_indicator[0])
                loss = self.compute_loss(logits, support_label[0])

            stop_probability = 0
            if config.FLEXIBLE_STEP:
                stop_probability = self.stop(k, loss, score)
                self.stop_prob = stop_probability
            
            stop_gates.append(stop_probability)
            scores.append(score.item())

            with torch.no_grad():
                pred = F.softmax(logits, dim=1).argmax(dim=1)
                if not self.paper:
                    correct = torch.eq(pred, support_data.y).sum().item()
                    train_accs.append(correct / support_data.y.shape[0])
                else:
                    correct = torch.eq(pred, support_label[0]).sum().item()
                    train_accs.append(correct / support_label.size(0))

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
            
            if not self.paper:
                logits_q, _, _ = self.net(query_data.x, query_data.edge_index, query_data.batch)
                loss_q = self.compute_loss(logits_q, query_data.y)
            else:
                logits_q, _, _ = self.net(query_nodes[0], query_edge_index[0], query_graph_indicator[0])
                loss_q = self.compute_loss(logits_q, query_label[0])

            losses_q.append(loss_q)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                if not self.paper:
                    correct = torch.eq(pred_q, query_data.y).sum().item()
                else:
                    correct = torch.eq(pred_q, query_label[0]).sum().item()
                corrects.append(correct)
        
        final_loss = losses_q[step]
        accs = np.array(corrects) / (query_size)
        final_acc = accs[step]
        total_loss = 0

        if config.FLEXIBLE_STEP:
            for step, (stop_gate, step_acc) in enumerate(zip(stop_gates[config.MIN_STEP - 1:], accs[config.MIN_STEP - 1:])):
                assert stop_gate >= 0.0 and stop_gate <= 1.0, "stop_gate error value: {:.5f}".format(stop_gate)
                log_prob = torch.log(1 - stop_gate)
                tem_loss = - log_prob * ((final_acc - step_acc - (np.exp(step) - 1) * config.STEP_PENALITY))
                total_loss += tem_loss

            total_loss = (total_loss + final_acc + final_loss)
        else:
            total_loss = final_loss

        total_loss.backward()

        if self.task_index == config.BATCH_PER_EPISODES:
            if config.GRAD_CLIP > 0.1:
                torch.nn.utils.clip_grad_norm_(self.parameters(), config.GRAD_CLIP)

            self.meta_optim.step()
            self.meta_optim.zero_grad()
            self.task_index = 1
        else:
            self.task_index += 1
        
        if config.FLEXIBLE_STEP:
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
        
        ada_step = min(config.STEP_TEST, config.MIN_STEP + int(2 / self.stop_prob))

        for k in range(ada_step):
            if not self.paper:
                logits, score, _ = self.net(support_data.x, support_data.edge_index, support_data.batch)
                loss = self.compute_loss(logits, support_data.y)
            else:
                logits, score, _ = self.net(support_nodes[0], support_edge_index[0], support_graph_indicator[0])
                loss = self.compute_loss(logits, support_label[0])

            stop_probability = 0

            if config.FLEXIBLE_STEP:
                with torch.no_grad():
                    stop_probability = self.stop(k, loss, score)
            
            stop_gates.append(stop_probability)
            step = k
            scores.append(score.item())

            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []

            for index, weight in enumerate(self.net.parameters()):
                if weight.fast is None:
                    weight.fast = weight - config.INNER_LR * grad[index]
                else:
                    weight.fast = weight.fast - config.INNER_LR * grad[index]

                fast_parameters.append(weight.fast)

            if not self.paper:
                logits_q, _, graph_emb = self.net(query_data.x, query_data.edge_index, query_data.batch)
                self.graph_labels.append(query_data.y)
            else:
                logits_q, _, graph_emb = self.net(query_nodes[0], query_edge_index[0], query_graph_indicator[0])
                self.graph_labels.append(query_label[0])

            self.graph_embs.append(graph_emb)

            if self.index % 1 == 0:
                self.index = 1
                self.graph_embs = []
                self.graph_labels = []
            else:
                self.index += 1
            
            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                if not self.paper:
                    correct = torch.eq(pred_q, query_data.y).sum().item()
                else:
                    correct = torch.eq(pred_q, query_label[0]).sum().item()

                corrects.append(correct)

                if not self.paper:
                    loss_query = self.compute_loss(logits_q, query_data.y)
                else:
                    loss_query = self.compute_loss(logits_q, query_label[0])
                    
                query_loss.append(loss_query.item())

        accs = 100 * np.array(corrects) / query_size

        if config.FLEXIBLE_STEP:
            stop_gates = [stop_gate.item() for stop_gate in stop_gates]

        return accs, step, stop_gates, scores, query_loss