import torch
import torch.nn as nn
# import main
import argparse
import random
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import copy

from utils.data_utils import read_poison_data

class vallinaAttacker(object):
    def __init__(self, args):
        self.args = args
        self.test_poison_loader = None

        self.backdoor_steps = self.args.local_epochs
        self.trigger_num = 4
        self.poisoning_per_batch = 7
        self.trigger_pattern = {'0_poison_pattern':[[14, 14], [15, 14], 
                                                    [14, 15], [15, 15]], 
                                '1_poison_pattern':[[16, 14], [17, 14], 
                                                    [16, 15], [17, 15]], 
                                '2_poison_pattern':[[14, 16], [15, 16], 
                                                    [14, 17], [15, 17]], 
                                '3_poison_pattern':[[16, 16], [17, 16], 
                                                    [16, 17], [17, 17]]} 


    def load_test_poison_data(self, clientObj, batch_size=None):
        if batch_size == None:
            batch_size = clientObj.batch_size
        test_poison_data_ori = read_poison_data(clientObj.dataset)
        test_poison_loader = DataLoader(test_poison_data_ori, batch_size, drop_last=False, shuffle=True)
        
        X_test_poison_data, y_test_poison_data = None, None
        for i, (x, y) in enumerate(test_poison_loader):
            x, y = self.get_poison_batch((x, y), evaluation=True)
            if X_test_poison_data == None:
                X_test_poison_data = x.clone().detach()
                y_test_poison_data = y.clone().detach()
            else:
                X_test_poison_data = torch.cat([X_test_poison_data, x], dim=0)
                y_test_poison_data = torch.cat([y_test_poison_data, y], dim=0)

        test_poison_data = TensorDataset(X_test_poison_data, y_test_poison_data)
        self.test_poison_loader = DataLoader(test_poison_data, batch_size, drop_last=False, shuffle=True)


    def get_poison_batch(self, bptt, evaluation=False):
        images, targets = bptt

        new_images = copy.deepcopy(images.detach())
        new_targets = copy.deepcopy(targets.detach())

        for index in range(0, len(images)):
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
        
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        
        return new_images, new_targets


    def add_pixel_pattern(self, ori_image):
        image = copy.deepcopy(ori_image)
        for idx in range(0, len(self.trigger_pattern)):
            poison_pattern = self.trigger_pattern[str(idx)+'_poison_pattern']

            for pos in poison_pattern:
                if image.size(0) == 1:
                    image[0][pos[0]][pos[1]] = 1 
                else:
                    image[0][pos[0]][pos[1]] = 1
                    image[1][pos[0]][pos[1]] = 1
                    image[2][pos[0]][pos[1]] = 1

        return image

    def train(self, clientObj):
        
        train_loader = clientObj.load_train_data()
        optimizer = torch.optim.SGD(clientObj.model.parameters(), lr=clientObj.learning_rate)
        clientObj.model.train()
        ce_loss = clientObj.loss

        if clientObj.args.algorithm == "FedRep":
            optimizer = torch.optim.SGD(clientObj.model.parameters(), lr=clientObj.learning_rate)
            for name, param in clientObj.model.named_parameters():
                if name.split('.')[0] == 'fc':
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            for epoch in range(clientObj.plocal_epochs):
                for i, (x, y) in enumerate(train_loader):
                    if type(x) == type([]):
                        x[0] = x[0].to(clientObj.device)
                    else:
                        x = x.to(clientObj.device)
                    y = y.to(clientObj.device)
                    # x, y = self.get_poison_batch((x, y), evaluation=False) 
                    output = clientObj.model(x)
                    loss = ce_loss(output, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
            max_local_epochs = clientObj.local_epochs
            
            for name, param in clientObj.model.named_parameters():
                if name.split('.')[0] == 'fc':
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            for epoch in range(max_local_epochs):
                for i, (x, y) in enumerate(train_loader):
                    if type(x) == type([]):
                        x[0] = x[0].to(clientObj.device)
                    else:
                        x = x.to(clientObj.device)
                    y = y.to(clientObj.device)
                    x, y = self.get_poison_batch((x, y), evaluation=False) 
                    output = clientObj.model(x)
                    loss = ce_loss(output, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        else:
            for step in range(self.backdoor_steps):
                for i, (x, y) in enumerate(train_loader):
                    x = x.to(clientObj.device)
                    x, y = self.get_poison_batch((x, y), evaluation=False) 
                    y = y.to(clientObj.device)
                    optimizer.zero_grad()
                    output = clientObj.model(x)
                    loss = ce_loss(output, y)
                    loss.backward()
                    optimizer.step()