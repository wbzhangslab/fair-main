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
import time
import os

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self, p):
		super(AT, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am

class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.args = args
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name
        
        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
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
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

        return test_acc, test_num

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))
    
    def nad_tuning(self):
        self.model.train()
        max_local_epochs = self.local_epochs 
        trainloader = self.load_train_data()

        # construct teacher model
        tmodel = copy.deepcopy(self.model)
        t_optimizer = torch.optim.SGD(tmodel.parameters(), lr=self.learning_rate)
        criterionAT = AT(1)
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output_t = tmodel(x)
                loss = self.loss(output_t, y)
                t_optimizer.zero_grad()
                loss.backward()
                t_optimizer.step()

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                # student model
                outputs_s = self.model(x)
                features_out3 = list(self.model.children())[:-1]
                modelout3 = nn.Sequential(*features_out3)
                modelout3.to(self.args.device)
                activation3_s = modelout3(x)

                features_out2 = list(self.model.children())[:-2]
                modelout2 = nn.Sequential(*features_out2)
                modelout2.to(self.args.device)
                activation2_s = modelout2(x)

                features_out1 = list(self.model.children())[:-3]
                modelout1 = nn.Sequential(*features_out1)
                modelout1.to(self.args.device)
                activation1_s = modelout1(x)

                # teacher model
                outputs_t = tmodel(x)
                features_out3 = list(tmodel.children())[:-1]
                modelout3 = nn.Sequential(*features_out3)
                modelout3.to(self.args.device)
                activation3_t = modelout3(x)

                features_out2 = list(tmodel.children())[:-2]
                modelout2 = nn.Sequential(*features_out2)
                modelout2.to(self.args.device)
                activation2_t = modelout2(x)

                features_out1 = list(tmodel.children())[:-3]
                modelout1 = nn.Sequential(*features_out1)
                modelout1.to(self.args.device)
                activation1_t = modelout1(x)
                
                # compute loss
                cls_loss = self.loss(outputs_s, y)
                at1_loss = criterionAT(activation1_s, activation1_t.detach()) * self.args.beta1
                at2_loss = criterionAT(activation2_s, activation2_t.detach()) * self.args.beta2
                at3_loss = criterionAT(activation3_s, activation3_t.detach()) * self.args.beta3

                loss = at1_loss + at2_loss + at3_loss + cls_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def fst_tuning(self):
        self.model.train()
        max_local_epochs = self.local_epochs * 2
        trainloader = self.load_train_data()

        original_linear_norm = torch.norm(eval('self.model.fc.weight'))
        weight_mat_ori = eval('self.model.fc.weight.data.clone().detach()')

        params_list = []
        for name, params in self.model.named_parameters():
            if 'fc' in name:
                if init:
                    if 'weight' in name:
                        # print(f'Initialize linear classifier weight {name}.')
                        std = 1 / math.sqrt(params.size(-1)) 
                        params.data.uniform_(-std, std)
                        
                    else:
                        # print(f'Initialize linear classifier weight {name}.')
                        params.data.uniform_(-std, std)
            
            if self.args.feature_shift_tuning == True:
                params.requires_grad = True
                params_list.append(params)
        
        fst_optimizer = torch.optim.SGD(params_list, lr=self.learning_rate)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = torch.sum(eval('self.model.fc.weight') * weight_mat_ori) + self.loss(output, y)
                fst_optimizer.zero_grad()
                loss.backward()
                fst_optimizer.step()

                exec_str = 'self.model.fc.weight.data = self.model.fc.weight.data * original_linear_norm  / torch.norm(self.model.fc.weight.data)'
                exec(exec_str)

    def simple_tuning(self):
        self.model.train()
        max_local_epochs = self.local_epochs
        trainloader = self.load_train_data()
        st_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate/10)

        for name, params in self.model.named_parameters():
            if 'fc' not in name:
                params.requires_grad = False
            else:
                self.kaiming_init_layer(params, mode='fan_in', nonlinearity='relu')
                params.requires_grad = True
        
        for epoch in range(max_local_epochs * 5):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                st_optimizer.zero_grad()
                loss.backward()
                st_optimizer.step()

        for name, params in self.model.named_parameters():
            if 'fc' not in name:
                params.requires_grad = True

    def kaiming_init_layer(self, layer, mode='fan_in', nonlinearity='relu'):
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            # Kaiming 初始化权重
            init.kaiming_normal_(layer.weight, mode=mode, nonlinearity=nonlinearity)
            # 偏置初始化为 0（如果有）
            if layer.bias is not None:
                init.zeros_(layer.bias)
        
        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # BatchNorm 权重初始化为 1，偏置初始化为 0
            init.ones_(layer.weight)
            init.zeros_(layer.bias)
        
        elif isinstance(layer, (nn.LSTM, nn.GRU)):
            # 初始化 LSTM/GRU 的权重（可选 Kaiming）
            for name, param in layer.named_parameters():
                if 'weight_ih' in name:  # 输入到隐藏层的权重
                    init.kaiming_normal_(param, mode=mode, nonlinearity=nonlinearity)
                elif 'weight_hh' in name:  # 隐藏层到隐藏层的权重
                    init.orthogonal_(param)  # 推荐正交初始化
                elif 'bias' in name:
                    init.zeros_(param)  # 偏置初始化为 0
