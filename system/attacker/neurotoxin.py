import torch
import torch.nn as nn
import math
import argparse
import random
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import copy

from utils.data_utils import read_poison_data

class neurotoxinAttacker(object):
    def __init__(self, args):
        self.args = args
        if args.algorithm == "FedFSR":
            self.attack_model = args.attack_model
        self.test_poison_loader = None

        self.backdoor_steps = self.args.local_epochs
        self.trigger_num = 4
        self.poisoning_per_batch = 7
        self.malicious_aggregate_all_layer = False
        self.poisoned_projection_norm = 5
        self.malicious_neurotoxin_ratio = 0.99
        
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

    def load_global_params(self, global_params):
        self.global_params = global_params

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

    def grad_mask_cv(self, clientObj, train_loader, ratio=None):
        """Generate a gradient mask based on the given dataset"""
        if self.args.algorithm == "FedFSR":
            model = self.attack_model
        else:
            model = clientObj.model

        model.train()
        model.zero_grad()

        for internal_round in range(10):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(clientObj.device), labels.to(clientObj.device)
                
                output = model(inputs)
                loss = nn.functional.cross_entropy(output, labels)
                loss.backward(retain_graph=True)
        mask_grad_list = []

        if self.malicious_aggregate_all_layer == True:
            grad_list = []
            grad_abs_sum_list = []
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_list.append(parms.grad.abs().view(-1))
                    grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())

            grad_list = torch.cat(grad_list).to(clientObj.device)
            if not isinstance(ratio, list):
                _, indices = torch.topk(-1 * grad_list, int(len(grad_list)*ratio))
                mask_flat_all_layer = torch.zeros(len(grad_list)).to(clientObj.device)
                mask_flat_all_layer[indices] = 1.0

            else:
                left_ratio = ratio[0]
                right_ratio = ratio[1]
                _, left_indices = torch.topk(grad_list, int(len(grad_list)*left_ratio))
                _, right_indices = torch.topk(grad_list, int(len(grad_list)*right_ratio))
                mask_flat_all_layer = torch.zeros(len(grad_list)).to(clientObj.device)
                mask_flat_all_layer[right_indices] = 1.0
                mask_flat_all_layer[left_indices] = 0.0

            count = 0
            percentage_mask_list = []
            k_layer = 0
            grad_abs_percentage_list = []
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients_length = len(parms.grad.abs().view(-1))

                    mask_flat = mask_flat_all_layer[count:count + gradients_length ].to(clientObj.device)
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).to(clientObj.device))

                    count += gradients_length
                    percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0
                    percentage_mask_list.append(percentage_mask1)
                    grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))
                    k_layer += 1

        else:
            grad_abs_percentage_list = []
            grad_res = []
            l2_norm_list = []
            sum_grad_layer = 0.0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_res.append(parms.grad.view(-1))
                    l2_norm_l = torch.norm(parms.grad.view(-1).clone().detach().to(clientObj.device))/float(len(parms.grad.view(-1)))
                    l2_norm_list.append(l2_norm_l)
                    sum_grad_layer += l2_norm_l.item()

            grad_flat = torch.cat(grad_res)

            percentage_mask_list = []
            k_layer = 0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients = parms.grad.abs().view(-1)
                    gradients_length = len(gradients)
                    if ratio == 1.0:
                        _, indices = torch.topk(-1*gradients, int(gradients_length*1.0))
                    else:

                        ratio_tmp = 1 - l2_norm_list[k_layer].item() / sum_grad_layer
                        _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))

                    mask_flat = torch.zeros(gradients_length)
                    mask_flat[indices.cpu()] = 1.0
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())
                    percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0
                    percentage_mask_list.append(percentage_mask1)
                    k_layer += 1

        model.zero_grad()
        return mask_grad_list

    def apply_grad_mask(self, model, mask_grad_list):
        mask_grad_list_copy = iter(mask_grad_list)
        for name, parms in model.named_parameters():
            if parms.requires_grad:
                parms.grad = parms.grad * next(mask_grad_list_copy)

    def model_dist_norm(self, model, global_model_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - global_model_params[name].data, 2))
        return math.sqrt(squared_sum)

    def projection(self, clientObj, global_model_params):
        model_norm = self.model_dist_norm(clientObj.model, global_model_params)

        if model_norm > self.poisoned_projection_norm:
            norm_scale = self.poisoned_projection_norm / model_norm
            for name, param in clientObj.model.named_parameters():
                clipped_difference = norm_scale * (param.data - global_model_params[name])
                param.data.copy_(global_model_params[name]+clipped_difference)

        return True

    def train(self, clientObj, attack=None):
        
        train_loader = clientObj.load_train_data()
        optimizer = torch.optim.SGD(clientObj.model.parameters(), lr=clientObj.learning_rate)
        clientObj.model.train()
        ce_loss = clientObj.loss
        mask_grad_list = self.grad_mask_cv(clientObj=clientObj, train_loader=train_loader, 
                                            ratio=self.malicious_neurotoxin_ratio)
        
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
                    self.apply_grad_mask(self.attack_model, mask_grad_list)
                    optimizer.step()

            for name, param in clientObj.model.named_parameters():
                if name.split('.')[0] not in ['aux', 'separation', 'recalibration']:
                    param = local_model.state_dict()[name].clone()

        elif clientObj.args.algorithm == "FedRep":
            optimizer = torch.optim.SGD(clientObj.model.parameters(), lr=clientObj.learning_rate)
            for name, param in clientObj.model.named_parameters():
                if name.split('.')[0] == 'fc':
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            for epoch in range(clientObj.local_epochs):
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
                    self.apply_grad_mask(clientObj.model, mask_grad_list)
                    optimizer.step()
                    # self.projection(clientObj, self.global_params)
        
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
                    self.apply_grad_mask(clientObj.model, mask_grad_list)
                    optimizer.step()
                    # self.projection(clientObj, self.global_params)