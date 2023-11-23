import argparse
import ast
import os
import random
import sys
import time
from time import time

import dill

sys.path.insert(0, "./")

import matplotlib.pyplot as plt
import my_optimizers
import pandas as pd
import possible_defenses
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from datasets import get_dataset
from models import model_sets
from my_utils import utils
from sklearn.manifold import TSNE

plt.switch_backend('agg')

D_ = 2 ** 13
BATCH_SIZE = 1000

from torch.utils.data.sampler import SequentialSampler


class MySequentialSampler(SequentialSampler):
    def __init__(self, data_source):
        self.data_source = data_source
 
    def __iter__(self):
        return iter(self.data_source)
 
    def __len__(self):
        return len(self.data_source)


def set_train_loader():
    dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset)
    train_dataset = dataset_setup.get_transformed_dataset(args.path_dataset, None, True)
    
    all_sample_ids = list(range(len(train_dataset)))
    random.shuffle(all_sample_ids)

    if args.dataset == 'Criteo':
        train_loader = train_dataset
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size, 
            shuffle=False,
            sampler=MySequentialSampler(all_sample_ids)
            # num_workers=args.workers
        )

    # check size_bottom_out and num_classes
    if args.use_top_model is False:
        if dataset_setup.size_bottom_out != dataset_setup.num_classes:
            raise Exception('If no top model is used,'
                            ' output tensor of the bottom model must equal to number of classes.')

    return train_loader, all_sample_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    # dataset paras
    parser.add_argument('-d', '--dataset', default='Criteo', type=str,
                        help='name of dataset',
                        choices=['CIFAR10', 'CIFAR100', 'TinyImageNet', 'CINIC10L', 'Yahoo', 'Criteo', 'BCW'])
    parser.add_argument('--path-dataset', help='path_dataset',
                        type=str, default='D:/Datasets/yahoo_answers_csv/')
    # framework paras
    parser.add_argument('--use-top-model', help='vfl framework has top model or not. If no top model'
                                                'is used, automatically turn on direct label inference attack,'
                                                'and report label inference accuracy on the training dataset',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--test-upper-bound', help='if set to True, test the upper bound of our attack: if all the'
                                                   'adversary\'s samples are labeled, how accurate is the adversary\'s '
                                                   'label inference ability?',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--half', help='half number of features, generally seen as the adversary\'s feature num. '
                                       'You can change this para (lower that party_num) to evaluate the sensitivity '
                                       'of our attack -- pls make sure that the model to be resumed is '
                                       'correspondingly trained.',
                        type=int,
                        default=16)  # choices=[16, 14, 32, 1->party_num]. CIFAR10-16, Liver-14, TinyImageNet-32
    # evaluation & visualization paras
    parser.add_argument('--k', help='top k accuracy',
                        type=int, default=5)
    parser.add_argument('--if-cluster-outputsA', help='if_cluster_outputsA',
                        type=ast.literal_eval, default=True)
    # attack paras
    parser.add_argument('--use-mal-optim',
                        help='whether the attacker uses the malicious optimizer',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--use-mal-optim-all',
                        help='whether all participants use the malicious optimizer. If set to '
                             'True, use_mal_optim will be automatically set to True.',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--use-mal-optim-top',
                        help='whether the server(top model) uses the malicious optimizer',
                        type=ast.literal_eval, default=False)
    # saving path paras
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models and csv files',
                        default='./saved_experiment_results', type=str)
    # possible defenses on/off paras
    parser.add_argument('--ppdl', help='turn_on_privacy_preserving_deep_learning',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--gc', help='turn_on_gradient_compression',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--lap-noise', help='turn_on_lap_noise',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--multistep_grad', help='turn on multistep-grad',
                        type=ast.literal_eval, default=False)
    # paras about possible defenses
    parser.add_argument('--ppdl-theta-u', help='theta-u parameter for defense privacy-preserving deep learning',
                        type=float, default=0.75)
    parser.add_argument('--gc-preserved-percent', help='preserved-percent parameter for defense gradient compression',
                        type=float, default=0.75)
    parser.add_argument('--noise-scale', help='noise-scale parameter for defense noisy gradients',
                        type=float, default=1e-3)
    parser.add_argument('--multistep_grad_bins', help='number of bins in multistep-grad',
                        type=int, default=6)
    parser.add_argument('--multistep_grad_bound_abs', help='bound of multistep-grad',
                        type=float, default=3e-2)
    # training paras
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of datasets loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')  # TinyImageNet=5e-2, Yahoo=1e-3
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--step-gamma', default=0.1, type=float, metavar='S',
                        help='gamma for step scheduler')
    parser.add_argument('--stone1', default=50, type=int, metavar='s1',
                        help='stone1 for step scheduler')
    parser.add_argument('--stone2', default=85, type=int, metavar='s2',
                        help='stone2 for step scheduler')
    args = parser.parse_args()
    if args.use_mal_optim_all:
        args.use_mal_optim = True


    train_loader, all_sample_ids = set_train_loader()
    dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset)
    train_dataset = dataset_setup.get_transformed_dataset(args.path_dataset, None, True)
    # python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset ./data/CIFAR10  --k 4 --epochs 1 --half 16
    # python test.py -d CIFAR10 --path-dataset ./data/CIFAR10

    for batch_idx, (data, target) in enumerate(train_loader):
        start, end = batch_idx*args.batch_size, min((batch_idx+1)*args.batch_size, len(all_sample_ids))
        origin_sample_idx = all_sample_ids[start: end]
        if batch_idx == 1:
            print(torch.equal(data[21], train_dataset[origin_sample_idx[21]][0]))
            # print(all_sample_ids[0])
            # print(target[0])
            # print(data[0])
            # print(train_dataset[all_sample_ids[0]])
        elif batch_idx == 10:
            break
        

        