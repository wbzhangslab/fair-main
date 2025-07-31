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

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
from sklearn.model_selection import train_test_split

np.random.seed(1)
num_clients = 100
dir_path = "Cifar10/"

# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    test_poison_path = dir_path + "test_poison/"
    ood_path = dir_path + "ood/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return
        
    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Get Cifar100 OOD
    transform_ood = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 加载Cifar10 数据集
    trainset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)
    
    test_classes = {}
    for ind, x in enumerate(testset):
        _, label = x
        if label in test_classes:
            test_classes[label].append(ind)
        else:
            test_classes[label] = [ind]

    range_no_id = list(range(0, len(testset)))
    for image_ind in test_classes[1]:
        if image_ind in range_no_id:
            range_no_id.remove(image_ind)

    range_no_id = np.array(range_no_id)
    np.random.shuffle(range_no_id)
    range_no_id = np.random.choice(range_no_id, size=1000, replace=True)


    # 加载Cifar100 OOD
    oodset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata_ood", train=True, download=True, transform=transform_ood)
    
    indices = random.sample(range(len(oodset)), 800)

    oodloader = torch.utils.data.DataLoader(oodset, batch_size=800, 
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                                            drop_last=True)
    ood_datalist = list(oodloader)
    ood_datalist_shape = 800 // 800 * 800 
    
    ## 重新分配标签 [0,1,...,9] * 80  800 个标签 
    assigned_labels = np.array([i for i in range(10)] * (ood_datalist_shape//10) + [i for i in range(ood_datalist_shape%10)])
    np.random.shuffle(assigned_labels)
    assigned_labels = assigned_labels.reshape(800//800, 800) # 1, 800
    for batch_id, batch in enumerate(ood_datalist):
        data, targets = batch
        
        # 标签根据目标任务数据标签类别数，均匀分配给OOD数据集
        for ind in range(len(targets)):
            targets[ind] = assigned_labels[batch_id][ind]
    oodloader=iter(ood_datalist)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    for _, data_ood in enumerate(oodloader, 0):
        oodset.data, oodset.targets = data_ood


    dataset_image = []
    dataset_label = []

    dataset_ood_image = []
    dataset_ood_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    dataset_poison_image = []
    dataset_poison_label = []
    dataset_poison_image.extend(dataset_image[range_no_id])
    dataset_poison_label.extend(dataset_label[range_no_id])
    dataset_poison_image = np.array(dataset_poison_image)
    dataset_poison_label = np.array(dataset_poison_label)

    dataset_ood_image.extend(oodset.data.cpu().detach().numpy())
    dataset_ood_label.extend(oodset.targets.cpu().detach().numpy())
    
    dataset_ood_image = np.array(dataset_ood_image)
    dataset_ood_label = np.array(dataset_ood_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    num_classes_ood = len(set(dataset_ood_label))
    print(f'Number of classes(OOD): {num_classes_ood}')


    # 将Cifar10转为npz文件
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,  
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)
    
    if not os.path.exists(test_poison_path):
        os.makedirs(test_poison_path)
    
    X_test_poison = dataset_poison_image
    y_test_poison = dataset_poison_label

    test_poison = {'x': X_test_poison, 'y': y_test_poison}

    with open(test_poison_path + 'test_poison.npz', 'wb') as f:
        np.savez_compressed(f, data=test_poison)

    # 将OOD转为npz文件
    if not os.path.exists(ood_path):
        os.makedirs(ood_path)
    
    X_ood = dataset_ood_image
    y_ood = dataset_ood_label

    ood = {'x': X_ood, 'y': y_ood}

    with open(ood_path + 'ood.npz', 'wb') as f:
        np.savez_compressed(f, data=ood)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)