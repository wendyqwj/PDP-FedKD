import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os
import logging

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid,cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdateDP, LocalUpdateDPSerial, LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFemnist, CharLSTM
from models.Fed import FedAvg, FedWeightAvg
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from opacus.grad_sample import GradSampleModule
from datetime import datetime

if __name__ == '__main__':
    # parse args

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dict_users = {}
    dataset_train, dataset_test = None, None

    #log输出到文件
    rootpath = './log_ldp'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(rootpath + '/runlog_fed_{}_{}_T{}_N{}_iid{}_{}_epsilon{}_del{}_clip{}_C{}_E{}_B{}_{}.log'.format(args.dataset, args.model,
                                                                            args.epochs, args.num_users, args.iid,
                                                                            args.dp_mechanism, args.dp_epsilon, args.dp_delta, args.dp_clip,
                                                                            args.frac,args.local_ep, args.local_bs,
                                                                            datetime.now().strftime(
                                                                                '%Y-%m-%d-%H-%M-%S')), mode='w')
    fh.setLevel(logging.DEBUG)
    # 输出到终端
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        args.num_channels = 1
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        args.num_channels = 3
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion-mnist':
        args.num_channels = 1
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'femnist':
        args.num_channels = 1
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: femnist dataset is naturally non-iid')
        else:
            print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
    elif args.dataset == 'shakespeare':
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    net_glob = None
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == 'femnist' and args.model == 'cnn':
        net_glob = CNNFemnist(args=args).to(args.device)
    elif args.dataset == 'shakespeare' and args.model == 'lstm':
        net_glob = CharLSTM().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    # use opacus to wrap model to clip per sample gradient
    net_glob = GradSampleModule(net_glob)
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    all_clients = list(range(args.num_users))

    # training
    acc_test = []
    loss_train = []
    # 本地客户端加噪，或者不加噪
    if args.dp_mechanism != 'no_dp':
        if args.serial:
            clients = [LocalUpdateDPSerial(args=args, dataset=dataset_train, idxs=dict_users[i]) for i in
                       range(args.num_users)]
        else:
            clients = [LocalUpdateDP(args=args, dataset=dataset_train, idxs=dict_users[i]) for i in
                       range(args.num_users)]
    else:
        clients = [LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[i]) for i in
                   range(args.num_users)]

    m, loop_index = max(int(args.frac * args.num_users), 1), int(1 / args.frac)
    t_start = time.time()
    for iter in range(args.epochs):
        w_locals, loss_locals, weight_locols = [], [], []
        # round-robin selection
        begin_index = (iter % loop_index) * m
        end_index = begin_index + m
        idxs_users = all_clients[begin_index:end_index]
        for idx in idxs_users:
            local = clients[idx]
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            weight_locols.append(len(dict_users[idx]))

        # update global weights
        if args.avg:
            w_glob, bytes_overhead_comm = FedAvg(w_locals)
        else:
            w_glob, bytes_overhead_comm = FedWeightAvg(w_locals, weight_locols)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        if len(loss_locals):
            loss_avg = sum(loss_locals) / len(loss_locals)
        else:
            loss_avg = -1  # error state
        loss_train.append(loss_avg)

        # print accuracy
        net_glob.eval()
        # acc_t, loss_t = test_img(net_glob, dataset_test, args)
        acc_t, loss_t = test_img(net_glob, dataset_train, args)
        acc_test.append(acc_t.item())

        # log print
        t_end = time.time()
        logger.info("Round {:3d} \t Average Loss {:.3f} \t Testing accuracy: {:.2f} \t CommOverhead_PerIter: {}types \t Time taken: {:.2f}s".format(
            iter, loss_avg, acc_t, bytes_overhead_comm, t_end - t_start))

    # total log info
    logger.info("\n========TRAIN END========\n")

    logger.debug(
        "\n===================DATA Round 0-{}====================".format(iter))
    logger.debug("Test Accuracy: {}".format(acc_test))
    logger.debug("Train Loss: {}".format(loss_train))
    logger.debug(
        "===================END Round 0-{}====================".format(iter))

    # plot loss curve
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('Test Accuracy')
    plt.savefig(rootpath + '/accfig_fed_{}_{}_T{}_iid{}_{}_epsilon{}_del{}_clip{}_C{}_E{}_B{}_{}.png'.format(
        args.dataset, args.model, args.epochs, args.iid, args.dp_mechanism, args.dp_epsilon, args.dp_delta, args.dp_clip,
        args.frac, args.local_ep, args.local_bs, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('Train Loss')
    plt.savefig(rootpath + '/lossfig_fed_{}_{}_T{}_iid{}_{}_epsilon{}_del{}_clip{}_C{}_E{}_B{}_{}.png'.format(
        args.dataset, args.model, args.epochs, args.iid, args.dp_mechanism, args.dp_epsilon, args.dp_delta,
        args.dp_clip,
        args.frac, args.local_ep, args.local_bs, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
