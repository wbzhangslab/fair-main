# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import copy
import random
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data


class clientFAIR(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.args = args
        self.eps = args.eps
        self.alpha = args.alpha
        self.lam_sep = args.lam_sep
        self.lam_rec = args.lam_rec
        self.rand_percent = args.rand_percent
        
        self.trainloader = self.load_train_data()
        self.train_data = read_client_data(self.dataset, self.id, is_train=True)

    def get_pred(self, out, labels):
        pred = out.sort(dim=-1, descending=True)[1][:, 0] 
        second_pred = out.sort(dim=-1, descending=True)[1][:, 1] 
        adv_label = torch.where(pred == labels, second_pred, pred) 

        return adv_label

    def set_parameters(self, model):
        for (new_name, new_param), (old_name, old_param) in zip(model.named_parameters(), self.model.named_parameters()):
            if old_name.split('.')[0] not in ['aux', 'separation', 'recalibration']:
                old_param.data = new_param.data.clone()

    def train(self):
        start_time = time.time()
        max_local_epochs = self.local_epochs

        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio*len(self.train_data))
        rand_idx = random.randint(0, len(self.train_data)-rand_num)
        rand_loader = DataLoader(self.train_data[rand_idx:rand_idx+rand_num], self.batch_size)

        for name, param in self.model.named_parameters():
            if name.split('.')[0] not in ['aux', 'separation', 'recalibration']: 
                param.requires_grad = False
            else:
                param.requires_grad = True

        for epoch in range(self.args.local_epochs):
            for i, (x, y) in enumerate(self.trainloader): # rand_loader
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                # self.model.eval()
                self.model.train()
                adv_outputs, adv_r_outputs, adv_nr_outputs, adv_rec_outputs = self.model(x)
                adv_y = self.get_pred(adv_outputs, y)
                cls_loss = self.loss(adv_outputs, y)
                r_loss = torch.tensor(0.).to(self.device)
                if not len(adv_r_outputs) == 0:
                    for r_out in adv_r_outputs:
                        r_loss += self.lam_sep * self.loss(r_out, y)
                    r_loss /= len(adv_r_outputs)

                nr_loss = torch.tensor(0.).to(self.device)
                if not len(adv_nr_outputs) == 0:
                    for nr_out in adv_nr_outputs:
                        nr_loss += self.lam_sep * self.loss(nr_out, adv_y)
                    nr_loss /= len(adv_nr_outputs)
                sep_loss = r_loss + nr_loss

                rec_loss = torch.tensor(0.).to(self.device)
                if not len(adv_rec_outputs) == 0:
                    for rec_out in adv_rec_outputs:
                        rec_loss += self.lam_rec * self.loss(rec_out, y)
                    rec_loss /= len(adv_rec_outputs)
                loss = cls_loss + sep_loss + rec_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        for name, param in self.model.named_parameters():
            if name.split('.')[0] not in ['aux', 'separation', 'recalibration']: 
                param.requires_grad = True
            else:
                param.requires_grad = False

        for epoch in range(self.args.local_epochs):
            for i, (x, y) in enumerate(self.trainloader): 
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                    
                y = y.to(self.device)
                outputs, _, _, _ = self.model(x)
                cls_loss = self.loss(outputs, y)
                loss = cls_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output, _, _, _ = self.model(x, is_eval=True)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

        return test_acc, test_num