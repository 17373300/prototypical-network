import json
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizer
import random
from mini_imagenet import PNDataset, my_collate
import torch.nn as nn
from utils import pprint, set_gpu, ensure_path, count_acc, euclidean_metric


trainset = PNDataset('rob')
from samplers import CategoriesSampler
from convnet import PNModel

train_sampler = CategoriesSampler(trainset.label, 2, 1)
train_loader = DataLoader(dataset=trainset,
                          batch_sampler=train_sampler,
                          collate_fn=my_collate)
criterion = nn.CrossEntropyLoss()

for i, batch in enumerate(train_loader):
    sen, label = batch
    # print(sen)
    # print(label)
    model = PNModel()
    support = model(sen)
    # print(support[0,:])
    # print(torch.sum(label, 0))

    # label = label.T.unsqueeze(2).repeat((1, 1, support.size(1)))
    # support = support.unsqueeze(0).repeat((13, 1, 1))
    # proto =label * support
    print(euclidean_metric(support, support, 1, 13, label))
    # print(proto.size())
    # print(label[0, :, :])
    # print(proto[0, :, :])

    # proto = support.reshape(1, 13, -1).mean(dim=0)
    # print(proto.size())
    # support_norm = torch.norm(support, p=2, dim=1, keepdim=True)
    # proto_norm = torch.norm(proto, p=2, dim=1, keepdim=True)
    # norm = torch.matmul(support_norm, proto_norm.T)

    # sim = torch.matmul(support, proto.T)
    # print(sim.size())
    # print(norm.size())
    # sim = sim / norm
    # print(sim[1])
    # print(label[1])
    # print(criterion(sim.float(), label.float()))
    break