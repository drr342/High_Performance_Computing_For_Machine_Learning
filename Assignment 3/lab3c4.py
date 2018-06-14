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

    rank = dist.get_rank()
    size = dist.get_world_size() - 1
    # rank_batch = int(BATCH_SIZE / float(size))
    rank_batch = BATCH_SIZE
    if (rank == MASTER):
        return int(math.ceil(len(labels) / (rank_batch * size))), torch.zeros(1)

    labels = torch.LongTensor(labels)
    resize = Transforms.Resize((RESIZE, RESIZE))
    to_tensor = Transforms.ToTensor()
    preprocess = Transforms.Compose([resize, to_tensor])
    images = MyImageFolder(path,
                           transform=preprocess,
                           target_transform=lambda i: labels[i])

    rng = Random()
    rng.seed(1234)
    index_train = list(range(len(df_train)))
    rng.shuffle(index_train)
    index_train = list(map(list, np.array_split(index_train, size)))
    index_train = index_train[rank - 1]

    train_sampler = Sampler.SubsetRandomSampler(index_train)
    train_data_loader = DataLoader(images,
                                   batch_size=rank_batch,
                                   shuffle=False,
                                   sampler=train_sampler,
                                   num_workers=WORKERS)
    return train_data_loader, len(index_train)


def run():
    size = dist.get_world_size() - 1
    rank = dist.get_rank()

    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    model.train()
    train_set, samples = create_data_loader()

    if (rank != MASTER):
        # for param in model.parameters():
        #     dist.broadcast(param.data, MASTER)
        start = monotonic()
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            for data, target in train_set:
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = model(data)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(output, target)
                # if (rank == 1):
                #     print(loss)
                epoch_loss += loss
                loss.backward()
                for param in model.parameters():
                    dist.send(param.grad.data, 0)
                for param in model.parameters():
                    dist.recv(param.data, 0)
            dist.barrier()
            dist.broadcast(param.data, MASTER)
        end = monotonic()
        epoch_loss *= samples / (math.ceil(samples / BATCH_SIZE))
        dist.reduce(epoch_loss, MASTER, op=dist.reduce_op.SUM)
        dist.reduce(torch.Tensor([samples]), MASTER, op=dist.reduce_op.SUM)
        dist.reduce(torch.Tensor([end-start]), MASTER, op=dist.reduce_op.SUM)
    else:
        for param in model.parameters():
            dummy_loss = 0
            for this_p in model.parameters():
                dummy_loss += torch.mean(this_p)
            dummy_loss.backward()
            # dist.broadcast(param.data, MASTER)
        for epoch in range(EPOCHS):
            # print('Rank {}, Epoch {} of {}...'.format(rank, epoch + 1, EPOCHS))
            for j in range(train_set * size):
                # print('{} of {}'.format(j, train_set * size), end = '\r')
                for i, param in enumerate(model.parameters()):
                    if (i == 0):
                        r = dist.recv(param.grad.data)
                    else:
                        dist.recv(param.grad.data, r)
                optimizer.step()
                for param in model.parameters():
                    dist.send(param.data, r)
            dist.barrier()
            dist.broadcast(param.data, MASTER)
        epoch_loss = torch.zeros(1)
        avg_time = torch.zeros(1)
        dist.reduce(epoch_loss, MASTER, op=dist.reduce_op.SUM)
        dist.reduce(samples, MASTER, op=dist.reduce_op.SUM)
        dist.reduce(avg_time, MASTER, op=dist.reduce_op.SUM)
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
