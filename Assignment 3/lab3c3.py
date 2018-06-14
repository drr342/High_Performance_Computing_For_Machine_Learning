# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 12:04:48 2018

@author: drr342
"""

from __future__ import print_function, division
import pandas as pd
import torch
import numpy as np
from random import Random
import torch.distributed as dist
import torchvision.datasets as Datasets
import torchvision.transforms as Transforms
import torch.utils.data.sampler as Sampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from time import monotonic
import math
import sys

RESIZE = 32
BATCH_SIZE = 100
INPUT_DIM = RESIZE * RESIZE * 3
L1_DIM = 1024
L2_DIM = 256
OUT_DIM = 17
LR = 0.01
MOMENTUM = 0.9
EPOCHS = 5
WORKERS = 1
MASTER = 0


class MyImageFolder(Datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        target = self.target_transform(index)
        img = self.transform(img)
        return img, target


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(INPUT_DIM, L1_DIM)
        self.linear2 = nn.Linear(L1_DIM, L2_DIM)
        self.linear3 = nn.Linear(L2_DIM, OUT_DIM)

    def forward(self, input):
        x = input.view(-1, INPUT_DIM)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def create_data_loader():
    path = '/scratch/am9031/CSCI-GA.3033-023/lab1/kaggleamazon/'
    df_train = pd.read_csv(path + 'train.csv')
    labels = df_train[df_train.columns[1]].tolist()
    labels = torch.LongTensor(labels)

    resize = Transforms.Resize((RESIZE, RESIZE))
    to_tensor = Transforms.ToTensor()
    preprocess = Transforms.Compose([resize, to_tensor])
    images = MyImageFolder(path,
                           transform=preprocess,
                           target_transform=lambda i: labels[i])

    rank = dist.get_rank()
    size = dist.get_world_size()
    # rank_batch = int(BATCH_SIZE / float(size))
    rank_batch = BATCH_SIZE
    rng = Random()
    rng.seed(1234)
    index_train = list(range(len(df_train)))
    rng.shuffle(index_train)
    index_train = list(map(list, np.array_split(index_train, size)))
    index_train = index_train[rank]

    train_sampler = Sampler.SubsetRandomSampler(index_train)
    train_data_loader = DataLoader(images,
                                   batch_size=rank_batch,
                                   shuffle=False,
                                   sampler=train_sampler,
                                   num_workers=WORKERS)
    return train_data_loader, len(index_train)


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def run():
    size = dist.get_world_size()
    rank = dist.get_rank()

    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    model.train()
    train_set, samples = create_data_loader()

    start = monotonic()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        # for i, (data, target) in enumerate(train_set):
        for data, target in train_set:
            # if (rank == MASTER):
            #     print('Epoch: {}, Minibatch: {}'.format(
            #         epoch + 1, i + 1), end='\r')
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            epoch_loss += loss
            loss.backward()
            average_gradients(model)
            optimizer.step()
        # if (rank == MASTER):
        #     print()
    end = monotonic()
    avg_time = torch.Tensor([end-start])
    epoch_loss *= samples / (math.ceil(samples / BATCH_SIZE))
    samples = torch.Tensor([samples])
    dist.reduce(epoch_loss, MASTER, op=dist.reduce_op.SUM)
    dist.reduce(samples, MASTER, op=dist.reduce_op.SUM)
    dist.reduce(avg_time, MASTER, op=dist.reduce_op.SUM)
    if (rank == MASTER):
        # print(float(samples))
        # print(float(epoch_loss))
        epoch_loss = float(epoch_loss) / float(samples)
        avg_time = float(avg_time) / (EPOCHS * size)
        print('{:.4f}, {:.4f}'.format(avg_time, epoch_loss))


def main():
    dist.init_process_group(backend='mpi')
    run()


if __name__ == "__main__":
    main()
