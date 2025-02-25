import copy
import pickle
import torch
import sys
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedWeightAvg(w, size, names, params):
    w_size = sys.getsizeof(pickle.dumps(w)) 
    print("w_size = {}".format(w_size))
    totlen = 0
    totalSize = sum(size)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        # print(w_avg[k])
        w_avg[k] = torch.div(w_avg[k], totalSize)
        totlen += len(w_avg[k])
    sum_params = {}
    sub_params = {}
    dot_params = {}
    var_params = {}

    param_len = len(params)
    print('param_len: ', param_len)

    for i in range(len(names[0])):
        sum_params[names[0][i]] = torch.zeros_like(params[0][i])
        sub_params[names[0][i]] = torch.zeros_like(params[0][i])
        dot_params[names[0][i]] = torch.zeros_like(params[0][i])

    # get sum of gradients of all clients in each layer
    for i in range(param_len):
        for j in range(len(params[i])):
            sum_params[names[0][j]] += params[i][j]

    # get mean of all clients (\mu_{i,j})
    for i in range(len(params[0])):
        sum_params[names[0][i]] /= param_len
        # print('name: ', names[0][i], ', sum: ', sum_params[names[0][i]])



  
    # get sum of squares of differences ((\sum_k (w_{i,j}^k - \mu_{i,j})^2) / K)
    for i in range(param_len):
        for j in range(len(params[i])):
            # print('shapes: ', params[i][j].shape, sum_params[names[0][j]].shape)
            sub_params[names[0][j]] += torch.sub(params[i][j], sum_params[names[0][j]])
            # 取消改行注释，输出shape
            # print('shapes: ', sub_params[names[0][j]].shape)
            dot_params[names[0][j]] += torch.square(sub_params[names[0][j]]) / param_len

    # calculate v_c of each layer (\sum \sum VAR / (q * p))
    v_c = {}
    for i in range(len(names[0])):
        v_c[names[0][i]] = torch.sum(dot_params[names[0][i]]) / (len(params[0][i]) * param_len)
        print('name: ', names[0][i], ' -- v_c: ', v_c[names[0][i]])

    # print('sum: ', sum_params)
    # print('sub: ', sub_params)
    # print('dot: ', dot_params)
    # for name, param in params[0]:
    #     print('-->name:', name, '-->var: ', torch.var(param.grad))

   # g_avg = copy.deepcopy(grads[0])
   # for g in g_avg.keys():
   #     for i in range(1, len(grads)):
   #         g_avg[g] += grads[i][g]
   # print(g_avg)
    return w_avg
