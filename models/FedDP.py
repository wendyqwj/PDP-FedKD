#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from utils.dp_mechanism import cal_sensitivity, cal_sensitivity_MA, Laplace, Gaussian_Simple, Gaussian_MA
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
import pickle
import copy
import random
from sklearn import metrics

class ServerAggDP(object):
    def __init__(self, args):
        self.args =args
        self.times = self.args.epochs
        self.lr = args.lr
        self.noise_scale = self.calculate_noise_scale()
        self.select_num = max(int(args.frac * args.num_users), 1)

    def calculate_noise_scale(self):
        if self.args.dp_mechanism == 'Laplace':
            epsilon_single_query = self.args.dp_epsilon / self.times
            return Laplace(epsilon=epsilon_single_query)
        elif self.args.dp_mechanism == 'Gaussian':
            epsilon_single_query = self.args.dp_epsilon / self.times
            delta_single_query = self.args.dp_delta / self.times
            return Gaussian_Simple(epsilon=epsilon_single_query, delta=delta_single_query)
        elif self.args.dp_mechanism == 'MA':
            return Gaussian_MA(epsilon=self.args.dp_epsilon, delta=self.args.dp_delta, q=self.args.dp_sample,
                               epoch=self.times)

    def FedAvg(self, w):
        w_size = sys.getsizeof(pickle.dumps(w))
        # print("w_size = {}".format(w_size))
        # sum
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
        # add noise
        if self.args.dp_mechanism != 'no_dp':
            self.add_noise(w_avg)
        # avg
        for k in w_avg.keys():
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg, w_size

    def FedWeightAvg(self, w, size):
        w_size = sys.getsizeof(pickle.dumps(w))
        # print("w_size = {}".format(w_size))
        totalSize = sum(size)
        # weight sum
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            w_avg[k] = w[0][k] * size[0]
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k] * size[i]
        # add noise
        if self.args.dp_mechanism != 'no_dp':
            self.add_noise(w_avg)
        # avg
        for k in w_avg.keys():
            w_avg[k] = torch.div(w_avg[k], totalSize)
        return w_avg, w_size


    def add_noise(self, w):
        sensitivity = cal_sensitivity(self.lr, self.args.dp_clip, self.select_num)
        state_dict = w
        if self.args.dp_mechanism == 'Laplace':
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.laplace(loc=0, scale=sensitivity * self.noise_scale,
                                                                        size=v.shape)).to(self.args.device)
        elif self.args.dp_mechanism == 'Gaussian':
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
                                                                       size=v.shape)).to(self.args.device)
        elif self.args.dp_mechanism == 'MA':
            sensitivity = cal_sensitivity_MA(self.args.lr, self.args.dp_clip, self.select_num)
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
                                                                       size=v.shape)).to(self.args.device)