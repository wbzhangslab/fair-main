import torch
import torch.nn as nn
import argparse
import random
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import copy

from utils.data_utils import read_poison_data, read_client_data


class pfedbaAttacker:
    def __init__(self, args):
        self.args = args
        self.first_attack = False
        if args.algorithm == "FedFSR":
            self.attack_model = args.attack_model
        self.test_poison_loader = None

        self.noise_mask = None
        self.optimize_trigger = None

        self.backdoor_steps = self.args.local_epochs * 2  # 后门训练步数
        self.poisoning_per_batch = 7  # fashionmnist 16 other 7
        self.trigger_pattern = [
            [14, 14], [15, 14], [16, 14], [17, 14],  
            [14, 15], [15, 15], [16, 15], [17, 15],
            [14, 16], [15, 16], [16, 16], [17, 16], 
            [14, 17], [15, 17], [16, 17], [17, 17]                   
        ]

    def load_test_poison_data(self, clientObj, batch_size=None):
        if batch_size == None:
            batch_size = clientObj.batch_size
        test_poison_data_ori = read_poison_data(clientObj.dataset)
        test_poison_loader = DataLoader(test_poison_data_ori, batch_size, drop_last=False, shuffle=True)
        
        X_test_poison_data, y_test_poison_data = None, None
        for i, (x, y) in enumerate(test_poison_loader):
            x, y = self.get_poison_batch((x, y), evaluation=True)
            if X_test_poison_data == None:
                X_test_poison_data = x
                y_test_poison_data = y
            else:
                X_test_poison_data = torch.cat([X_test_poison_data, x], dim=0)
                y_test_poison_data = torch.cat([y_test_poison_data, y], dim=0)

        test_poison_data = TensorDataset(X_test_poison_data, y_test_poison_data)
        self.test_poison_loader = DataLoader(test_poison_data, batch_size, drop_last=False, shuffle=True)

    def init_trigger(self, clientObj):
        ## noise mask
        
        """初始化后门触发器"""
        train_loader = clientObj.load_train_data()
        for i, (x, y) in enumerate(train_loader):
            x = Variable(x.to(clientObj.device))
            sz = x.size()[1:]
            self.optimize_trigger = torch.zeros(sz).float()
            self.noise_mask = torch.ones(sz).float()
            for pos in self.trigger_pattern:
                if self.optimize_trigger.size(0) == 1:  
                    self.optimize_trigger[0, pos[0], pos[1]] = 0.5
                    self.noise_mask[0, pos[0], pos[1]] = 0
                elif self.optimize_trigger.size(0) == 3: 
                    for c in range(3):
                        self.optimize_trigger[c, pos[0], pos[1]] = 0.5
                        self.noise_mask[c, pos[0], pos[1]] = 0
            self.optimize_trigger = Variable(self.optimize_trigger, requires_grad=True).to(clientObj.device)
            self.noise_mask = Variable(self.noise_mask, requires_grad=True).to(clientObj.device)
            break

    def trigger_optimize(self, clientObj, malicious_clients):
        """触发器优化：实现梯度和损失对齐"""
        ce_loss = clientObj.loss
        if clientObj.args.algorithm == "FedFSR":
            for name, param in clientObj.model.named_parameters():
                if name.split('.')[0] not in ['aux', 'separation', 'recalibration']:
                    self.args.attack_model.state_dict()[name] = param.clone()
            local_model = copy.deepcopy(self.args.attack_model)
        else:
            local_model = copy.deepcopy(clientObj.model)

        dataset = []
        for labelindex in range(clientObj.args.num_classes):
            count = 1
            for malicious_client in malicious_clients:
                malicious_client_traindata = read_client_data(malicious_client.dataset, malicious_client.id, is_train=True)
                malicious_client_trainloader = DataLoader(malicious_client_traindata, len(malicious_client_traindata), shuffle=True)
                for X, Y in malicious_client_trainloader:
                    for i in range(len(X)):
                        if Y[i] == labelindex and count < 100:
                            dataset.append((X[i], Y[i]))
                            count += 1
                        if count >= 100:
                            break
                    if count >= 100:
                        break
                if count >= 100:
                    break

        data_iterator = DataLoader(dataset, batch_size=clientObj.batch_size, shuffle=True)                

        # 初始化触发器
        if self.optimize_trigger == None:
            self.init_trigger(clientObj)
        local_model.eval()
        if clientObj.args.algorithm == "FedRep":
            for param in local_model.parameters():
                param.requires_grad = True
        
        if not self.first_attack:
            # 阶段1：λ=0，损失对齐
            for epoch in range(3): # 50 3
                for i, (x, y) in enumerate(data_iterator):
                    x = Variable(x.to(clientObj.device))
                    y = Variable(y.to(clientObj.device))
                    y_target = torch.LongTensor(y.size()).fill_(1)
                    y_target = Variable(y_target, requires_grad=False).to(clientObj.device)

                    for count in range(len(y)):
                        for pos in self.trigger_pattern:
                            if x[count].size(0) == 1:  # 单通道
                                x[count][0, pos[0], pos[1]] = 0
                            elif x[count].size(0) == 3:  # 三通道
                                for c in range(3):
                                    x[count][c, pos[0], pos[1]] = 0


                    output = local_model((x + self.optimize_trigger).float())
                    loss1 = ce_loss(output, y_target)
                    local_model.zero_grad()
                    loss1.backward(retain_graph=True)

                    if self.optimize_trigger.grad is not None:
                        # self.optimize_trigger.grad.fill_(0)
                        self.optimize_trigger = self.optimize_trigger - self.optimize_trigger.grad * 0.1 

                    for m in range(x.size(2)):
                        for n in range(x.size(2)):
                            if [m, n] not in self.trigger_pattern:
                                if self.optimize_trigger.size(0) == 1:
                                    self.optimize_trigger[0, m, n] = 0
                                elif self.optimize_trigger.size(0) == 3:  # 三通道
                                    for c in range(3):
                                        self.optimize_trigger[c, m, n] = 0

                    self.optimize_trigger = torch.clamp(self.optimize_trigger, -1, 1)
                    self.optimize_trigger = Variable(self.optimize_trigger, requires_grad=True).to(clientObj.device)
                    self.first_attack = True

        
        # 阶段2：λ=1，梯度对齐
        for epoch in range(3): # 30 3
            for i, (x, y) in enumerate(data_iterator):
                x = Variable(x.to(clientObj.device))
                y = Variable(y.to(clientObj.device))
                y_target = torch.LongTensor(y.size()).fill_(1)
                y_target = Variable(y_target.to(clientObj.device), requires_grad=False)

                for count in range(len(y)):
                    for pos in self.trigger_pattern:
                        if x[count].size(0) == 1:  # 单通道
                            x[count][0, pos[0], pos[1]] = 0
                        elif x[count].size(0) == 3:  # 三通道
                            for c in range(3):
                                x[count][c, pos[0], pos[1]] = 0

                # 良性梯度
                output_c = local_model(x.float())
                classloss_c = ce_loss(output_c, y)
                grads_c = torch.autograd.grad(classloss_c, local_model.parameters(), create_graph=True)
                grads_c = [g.detach().clone() for g in grads_c]
                
                # 后门梯度
                output_p = local_model((x + self.optimize_trigger).float())
                classloss_p = ce_loss(output_p, y_target)
                grads_p = torch.autograd.grad(classloss_p, local_model.parameters(), create_graph=True)
                grads_p = [g.detach().clone() for g in grads_p]

                # 梯度对齐损失
                loss2 = self.match_l2_loss(grads_p, grads_c)
                local_model.zero_grad()
                loss2.backward(retain_graph=True)
                
                if self.optimize_trigger.grad is not None:
                    self.optimize_trigger = self.optimize_trigger - self.optimize_trigger.grad * 0.1 
                
                with torch.no_grad():
                    for m in range(x.size(2)):
                        for n in range(x.size(2)):
                            if [m, n] not in self.trigger_pattern:
                                if self.optimize_trigger.size(0) == 1:
                                    self.optimize_trigger[0, m, n] = 0
                                elif self.optimize_trigger.size(0) == 3:  # 三通道
                                    for c in range(3):
                                        self.optimize_trigger[c, m, n] = 0

                self.optimize_trigger = torch.clamp(self.optimize_trigger, -1, 1)
                self.optimize_trigger = Variable(self.optimize_trigger, requires_grad=True)

    def get_poison_batch(self, bptt, evaluation=False):
        """生成毒化批次数据"""
        images, targets = bptt
        new_images = images.detach().clone()
        new_targets = targets.detach().clone()

        for index in range(0, len(images)):
            # new_images[index].requires_grad = True
            if evaluation:  # poison all data when testing
                new_targets[index] = 1 # target label
                new_images[index] = self.add_pixel_pattern(images[index])
                
            else:  # poison part of data when training
                if index < self.poisoning_per_batch:
                    new_targets[index] = 1 # target label
                    new_images[index] = self.add_pixel_pattern(images[index])
                else:
                    new_images[index] = images[index]
                    new_targets[index] = targets[index]

        new_images = new_images
        new_targets = new_targets.long()
        
        # if evaluation:
        #     new_images.requires_grad_(False)
        #     new_targets.requires_grad_(False)

        return new_images, new_targets

    def add_pixel_pattern(self, ori_image):
        """添加后门触发器到图像中"""
        image = ori_image.clone()
        noise = self.optimize_trigger
        
        for pos in self.trigger_pattern:
            if image.size(0) == 1:  # 单通道
                image[0, pos[0], pos[1]] = noise[0, pos[0], pos[1]]
            elif image.size(0) == 3:  # 三通道
                for c in range(3):
                    image[c, pos[0], pos[1]] = noise[c, pos[0], pos[1]]
        
        return torch.clamp(image, -1, 1)

    def train(self, clientObj, attack=False):
        """本地后门训练"""
        train_loader = clientObj.load_train_data()
        ce_loss = clientObj.loss
        if self.args.algorithm == "FedFSR":
            local_model = self.attack_model
            for name, param in clientObj.model.named_parameters():
                if name.split('.')[0] not in ['aux', 'separation', 'recalibration']:
                    local_model.state_dict()[name] = param.clone()
                    local_model.state_dict()[name].requires_grad = True
            local_model.train()
            optimizer = torch.optim.SGD(local_model.parameters(), lr=clientObj.learning_rate)
            for step in range(self.backdoor_steps):
                for i, (x, y) in enumerate(train_loader):
                    if attack == True:
                        x, y = self.get_poison_batch((x, y), evaluation=False) 
                    x, y = x.to(clientObj.device), y.to(clientObj.device), 
                    optimizer.zero_grad()
                    output = local_model(x)
                    loss = ce_loss(output, y)
                    loss.backward()
                    optimizer.step()

            for name, param in clientObj.model.named_parameters():
                if name.split('.')[0] not in ['aux', 'separation', 'recalibration']:
                    param = local_model.state_dict()[name].clone()


        elif clientObj.args.algorithm == "FedRep":
            clientObj.model.train()
            for name, param in clientObj.model.named_parameters():
                if name.split('.')[0] == 'fc':
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            max_local_epochs = clientObj.local_epochs
            for epoch in range(max_local_epochs * 2):
                for i, (x, y) in enumerate(train_loader):
                    if type(x) == type([]):
                        x[0] = x[0].to(clientObj.device)
                    else:
                        x = x.to(clientObj.device)
                    y = y.to(clientObj.device)
                    # x, y = self.get_poison_batch((x, y), evaluation=False) 
                    output = clientObj.model(x)
                    loss = ce_loss(output, y)
                    clientObj.optimizer.zero_grad()
                    loss.backward()
                    clientObj.optimizer.step()
                    
            for name, param in clientObj.model.named_parameters():
                if name.split('.')[0] == 'fc':
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            for epoch in range(self.backdoor_steps):
                for i, (x, y) in enumerate(train_loader):
                    if type(x) == type([]):
                        x[0] = x[0].to(clientObj.device)
                    else:
                        x = x.to(clientObj.device)
                    y = y.to(clientObj.device)
                    x, y = self.get_poison_batch((x, y), evaluation=False) 
                    output = clientObj.model(x)
                    loss = ce_loss(output, y)
                    clientObj.optimizer.zero_grad()
                    loss.backward()
                    clientObj.optimizer.step()

        else:
            clientObj.model.train()
            for step in range(self.backdoor_steps):
                for i, (x, y) in enumerate(train_loader):
                    x, y = self.get_poison_batch((x, y), evaluation=False) 
                    x, y = x.to(clientObj.device), y.to(clientObj.device), 
                    clientObj.optimizer.zero_grad()
                    output = clientObj.model(x)
                    loss = ce_loss(output, y)
                    loss.backward()
                    clientObj.optimizer.step()

    def match_l2_loss(self, grads_p, grads_c):
        gw_c_vec = []
        gw_p_vec = []
        for ig in range(len(grads_c)):
            gw_c_vec.append(grads_c[ig].reshape(-1))
            gw_p_vec.append(grads_p[ig].reshape(-1))

        gw_c_vec = torch.cat(gw_c_vec, dim=0)
        gw_p_vec = torch.cat(gw_p_vec, dim=0)
        gw_c_vec.requires_grad = True
        gw_p_vec.requires_grad = True

        dis = torch.sqrt(torch.sum((gw_p_vec - gw_c_vec) ** 2))

        return dis