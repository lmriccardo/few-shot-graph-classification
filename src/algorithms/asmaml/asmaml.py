import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from models.stopcontrol import StopControl

import math


class AdaptiveStepMAML(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, model, config):
        """
        :param args:
        """
        super(AdaptiveStepMAML, self).__init__()
        print("Model: ada_meta")
        self.inner_lr = config["inner_lr"]
        self.n_way = config["train_way"]
        self.k_spt = config["train_shot"]
        self.k_qry = config["train_query"]
        self.clip = config["grad_clip"]
        self.net = model
        self.paper = config["paper"]

        self.task_index = 1
        self.task_num = config["batch_per_episodes"]

        self.flexible_step = config["flexible_step"]
        self.min_step=config["min_step"]
        if self.flexible_step:
            self.max_step=config["max_step"]
        else :
            self.max_step=self.min_step
        self.stop_prob=0.5
        self.update_step_test = config["step_test"]

        self.step_penalty=config["step_penalty"]
        self.use_score = config["use_score"]
        self.use_loss = config["use_loss"]

        stop_gate_para=[]
        if self.flexible_step:
            stop_input_size = 0
            if self.use_score:
                stop_input_size = stop_input_size + 1
            if self.use_loss:
                stop_input_size = stop_input_size + 1

            hidden_size = stop_input_size * 10
            self.stop_gate = StopControl(stop_input_size, hidden_size)
            stop_gate_para=self.stop_gate.parameters()

        self.meta_optim = optim.Adam(
            [{'params': self.net.parameters(), 'lr':config['outer_lr']},
             {'params': stop_gate_para, 'lr': config['stop_lr']}],
             lr=0.0001, weight_decay=config["weight_decay"])

        self.loss=nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optim, mode='min',
                                                                    factor=0.5, patience=config["patience"],
                                                                    verbose=True, min_lr=1e-5)
        self.graph_embs = []
        self.graph_labels = []
        self.index = 1

    def com_loss(self,logits,label):
        return self.loss(logits, label.long())

    def forward(self, support_data, query_data):
        
        if self.paper:
            (support_nodes, support_edge_index, support_graph_indicator, support_label) = support_data
            (query_nodes, query_edge_index, query_graph_indicator, query_label) = query_data

        task_num = 1

        querysz = query_data.y.shape[0] if not self.paper else query_label.size()[1]

        losses_q = []
        corrects = []
        stop_gates,scores=[],[]
        train_losses,train_accs=[],[]
        for i in range(task_num):

            fast_parameters = list(self.net.parameters())
            for weight in self.net.parameters():
                weight.fast = None
            step=0
            self.stop_prob=0.1 if self.stop_prob<0.1 else self.stop_prob
            ada_step=min(self.max_step,self.min_step+int(1.0/self.stop_prob))

            for k in range(0, ada_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                support_nodes_ = support_nodes[i] if self.paper else support_data.x
                support_edge_index_ = support_edge_index[i] if self.paper else support_data.edge_index
                support_graph_indicator_ = support_graph_indicator[i] if self.paper else support_data.batch
                support_label_ = support_label[i] if self.paper else support_data.y

                logits, score, _= self.net(support_nodes_, support_edge_index_, support_graph_indicator_)

                loss=self.com_loss(logits,support_label_)
                stop_pro=0
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                try:
                    if self.flexible_step:
                        stop_pro = self.stop(k, loss, score)
                        self.stop_prob = stop_pro
                except AssertionError:
                    print("Logits: ", logits)
                    print("Score: ", score)
                    print([param for param in self.parameters()])
                    print("Loss: ", loss)
                    print("Loss Grad: ", grad)

                    import sys
                    sys.exit(1)

                stop_gates.append(stop_pro)
                scores.append(score.item())
                with torch.no_grad():
                    pred = F.softmax(logits, dim=1).argmax(dim=1)
                    correct = torch.eq(pred, support_label).sum().item()
                    train_accs.append(correct / support_label.size(0))

                step = k
                train_losses.append(loss.item())
                # buiuld graph supld fport gradient of gradient

                fast_parameters = []
                for index, weight in enumerate(self.net.parameters()):
                    # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                    if weight.fast is None:
                        weight.fast = weight - self.inner_lr * grad[index]  # create weight.fast
                    else:
                        # create an updated weight.fast,
                        # note the '-' is not merely minus value, but to create a new weight.fast
                        weight.fast = weight.fast - self.inner_lr * grad[index]

                    # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
                    fast_parameters.append(weight.fast)

                query_nodes_ = query_nodes[i] if self.paper else query_data.x
                query_edge_index_ = query_edge_index[i] if self.paper else query_data.edge_index
                query_graph_indicator_ = query_graph_indicator[i] if self.paper else query_data.batch
                query_label_ = query_label[i] if self.paper else query_data.y
                logits_q, _ ,_= self.net(query_nodes_, query_edge_index_, query_graph_indicator_)
                # loss_q will be overwritten and just keep the loss_q on last update step.

                loss_q = self.com_loss(logits_q,query_label_)

                losses_q.append(loss_q)

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, query_label_).sum().item()  # convert to numpy
                    corrects.append(correct)

        final_loss = losses_q[step]

        accs = np.array(corrects) / (querysz * task_num)
        final_acc=accs[step]
        total_loss=0
    
        if self.flexible_step:
            for step, (stop_gate, step_acc) in enumerate(zip(stop_gates[self.min_step-1:], accs[self.min_step-1:])):
                assert stop_gate >= 0.0 and stop_gate <= 1.0, "stop_gate error value: {:.5f}".format(stop_gate)
                log_prob = torch.log(1 - stop_gate)
                tem_loss = -log_prob * ((final_acc - step_acc - (np.exp((step))-1) * self.step_penalty))
                total_loss = total_loss + tem_loss

        if not self.flexible_step:
            total_loss=final_loss/task_num
        else:
            total_loss=(total_loss+final_acc+final_loss)/task_num

        total_loss.backward()

        if self.task_index == self.task_num:
            if self.clip > 0.1:  # 0.1 threshold wether to do clip
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)

            self.meta_optim.step()
            self.meta_optim.zero_grad()
            self.task_index = 1
        else:
            self.task_index = self.task_index + 1
        if self.flexible_step:
            stop_gates=[stop_gate.item() for stop_gate in stop_gates]
        return accs*100,step,final_loss.item(),total_loss.item(),stop_gates,scores,train_losses,train_accs

    def stop(self, step,loss,node_score):
        stop_hx=None
        if self.flexible_step:
            if step < self.max_step:
                inputs=[]
                if self.use_loss:
                    inputs = inputs + [loss.detach()]
                if self.use_score:
                    score=node_score.detach()
                    inputs = inputs + [score]

                inputs = torch.stack(inputs, dim=0).unsqueeze(0)
                try:
                    inputs = self.smooth(inputs)[0]
                except AssertionError:
                    print(inputs)
                    print(loss)
                    raise AssertionError()

                # assert torch.sum(torch.isnan(inputs)) == 0, 'inputs has nan'
                stop_gate, stop_hx = self.stop_gate(inputs, stop_hx)
                assert torch.sum(torch.isnan(stop_gate)) == 0, 'stop_gate has nan'

                return stop_gate

        return loss.new_zeros(1, dtype=torch.float)

    def smooth(self,weight, p=10, eps=1e-10):
        weight_abs = weight.abs()
        less = (weight_abs < math.exp(-p)).type(torch.float)
        noless = 1.0 - less
        log_weight = less * -1 + noless * torch.log(weight_abs + eps) / p
        sign = less * math.exp(p) * weight + noless * weight.sign()
        # assert  torch.sum(torch.isnan(log_weight))==0,'stop_gate input has nan'
        if torch.sum(torch.isnan(log_weight)) != 0:
            print("Log weight: ", log_weight)
            print("Weight: ", weight)
            raise AssertionError('stop_gate input has nan')
        return log_weight, sign

    def finetuning(self, support_data, query_data):

        if self.paper:
            (support_nodes, support_edge_index, support_graph_indicator, support_label) = support_data
            (query_nodes, query_edge_index, query_graph_indicator, query_label) = query_data

        task_num = 1

        querysz = query_data.y.shape[0] if not self.paper else query_label.size()[1]

        # losses_q = [0 for _ in range(self.update_step_test)]  # losses_q[i] is the loss on step i
        corrects =[]
        step=0
        stop_gates,scores,query_loss=[],[],[]

        lista = []

        for i in range(task_num):

            fast_parameters = list(self.net.parameters())  # the first gradient calcuated in line 45 is based on original weight
            for weight in self.net.parameters():
                weight.fast = None

            # self.stop_prob=0.1 if self.stop_prob<0.1 else self.stop_prob
            ada_step = min(self.update_step_test, self.min_step + int(2 / self.stop_prob))


            for k in range(0, ada_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                support_nodes_ = support_nodes[i] if self.paper else support_data.x
                support_edge_index_ = support_edge_index[i] if self.paper else support_data.edge_index
                support_graph_indicator_ = support_graph_indicator[i] if self.paper else support_data.batch
                support_label_ = support_label[i] if self.paper else support_data.y
                logits,score,_ = self.net(support_nodes_, support_edge_index_, support_graph_indicator_)
                loss = self.com_loss(logits, support_label_)
                stop_pro=0
                try:
                    if self.flexible_step:
                        stop_pro = self.stop(k, loss, score)
                        self.stop_prob = stop_pro
                except AssertionError:
                    print("Logits: ", logits)
                    print("Score: ", score)
                    print([param for param in self.parameters()])
                    print("Loss: ", loss)
                    print("Loss Grad: ", loss.grad)

                    import sys
                    sys.exit(1)

                        # if k >= self.min_step and stop_pro-0.2 > random.random():
                        #     break
                stop_gates.append(stop_pro)
                step = k
                scores.append(score.item())
                # buiuld graph supld fport gradient of gradient
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                fast_parameters = []

                for index, weight in enumerate(self.net.parameters()):
                    # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py

                    if weight.fast is None:
                        weight.fast = weight - self.inner_lr * grad[index]  # create weight.fast
                    else:
                        # create an updated weight.fast,
                        # note the '-' is not merely minus value, but to create a new weight.fast
                        weight.fast = weight.fast - self.inner_lr * grad[index]

                    # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
                    fast_parameters.append(weight.fast)

                query_nodes_ = query_nodes[i] if self.paper else query_data.x
                query_edge_index_ = query_edge_index[i] if self.paper else query_data.edge_index
                query_graph_indicator_ = query_graph_indicator[i] if self.paper else query_data.batch
                query_label_ = query_label[i] if self.paper else query_data.y
                logits_q, _, graph_emb = self.net(query_nodes_, query_edge_index_, query_graph_indicator_)
                self.graph_labels.append(query_label_.reshape(-1))
                self.graph_embs.append(graph_emb)

                if self.index%1==0:
                    self.index=1
                    self.graph_embs = []
                    self.graph_labels = []
                else :
                    self.index=self.index+1

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, query_label_).sum().item()  # convert to numpy
                    corrects.append(correct)
                    loss_query=self.com_loss(logits_q,query_label_)
                    query_loss.append(loss_query.item())

                    lista = [pred_q.tolist(), query_label_.tolist()]

        accs = 100 * np.array(corrects) / querysz * task_num
        if self.flexible_step:
            stop_gates=[stop_gate.item() for stop_gate in stop_gates]
        return accs,step,stop_gates,scores,query_loss, lista

    def adapt_meta_learning_rate(self, loss):
        self.scheduler.step(loss)

    def get_meta_learning_rate(self):
        epoch_learning_rate = []
        for param_group in self.meta_optim.param_groups:
            epoch_learning_rate.append(param_group['lr'])
        return epoch_learning_rate[0]