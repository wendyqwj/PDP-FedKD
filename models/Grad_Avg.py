#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import torch
import sys
from torch import nn


def GradAvg(grad):
    grad_size = sys.getsizeof(pickle.dumps(grad))
    # print("w_size = {}".format(w_size))
    grad_avg = copy.deepcopy(grad[0])
    for k in grad_avg.keys():
        for i in range(1, len(grad)):
            grad_avg[k] += grad[i][k]
        grad_avg[k] = torch.div(grad_avg[k], len(grad))
    return grad_avg, grad_size


