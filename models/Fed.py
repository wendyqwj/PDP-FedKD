import copy
import pickle
import torch
import sys
from torch import nn


def FedAvg(w):
    w_size = sys.getsizeof(pickle.dumps(w))
    # print("w_size = {}".format(w_size))
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg, w_size


def FedWeightAvg(w, size):
    w_size = sys.getsizeof(pickle.dumps(w))
    # print("w_size = {}".format(w_size))
    totalSize = sum(size)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        w_avg[k] = torch.div(w_avg[k], totalSize)
    return w_avg, w_size
