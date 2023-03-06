import os
import shutil
import time
import random
import numpy as np
import pprint
import torch


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


'''def euclidean_metric(support, query, shot, train_way):
    proto = support.reshape(shot, train_way, -1).mean(dim=0)
    query_norm = torch.norm(query, p=2, dim=1, keepdim=True)
    proto_norm = torch.norm(proto, p=2, dim=1, keepdim=True)
    norm = torch.matmul(query_norm, proto_norm.T)
    sim = torch.matmul(query, proto.T)
    sim = sim / norm

    return sim'''


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def euclidean_metric(support, query, shot, train_way, label_support):
    # (s,w) -> (w,s) -> (w,s,1) -> (w,s,Hd)
    label_proto = label_support.T.unsqueeze(2).repeat((1, 1, support.size(1)))
    # (s,hd) -> (1,s,hd) -> (w,s,hd)
    support_proto = support.unsqueeze(0).repeat((train_way, 1, 1))
    # (w,s,Hd) * (w,s,hd) -> (w,s,hd)
    proto = label_proto * support_proto

    # (w,s,hd) -> (w,hd)
    proto = torch.sum(proto, 1)
    # (s,w) -> (w) -> (w,hd)
    label = torch.sum(label_support, 0)
    label = label.unsqueeze(1).repeat((1, support.size(1)))
    label = label+0.01
    proto = proto / label

    query_norm = torch.norm(query, p=2, dim=1, keepdim=True)
    proto_norm = torch.norm(proto, p=2, dim=1, keepdim=True)
    norm = torch.matmul(query_norm, proto_norm.T)
    sim = torch.matmul(query, proto.T)
    norm = norm+0.01
    sim = sim / norm

    return sim


def count_acc(predict, label):
    tp = fp = fn = tn = 0
    for i in range(len(predict)):
        for j in range(len(predict[0])):
            if predict[i][j] > 0.7:
                if label[i][j] > 0.5:
                    tp += 1
                else:
                    fp += 1
            else:
                if label[i][j] > 0.5:
                    fn += 1
                else:
                    tn += 1
    return tp, fp, fn, tn


class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2
