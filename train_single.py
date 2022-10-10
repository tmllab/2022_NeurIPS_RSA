#!/usr/bin/env python

import os
import math
import random
import argparse

import torch
import torch.optim
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10

from methods import create_method
from common.tools import knn_validate, getTime, contrastive_train
from common.augments import TwoCropsTransform, MultiStageTransform, GaussianBlur
from common.LmdbDataset import LmdbDataset
from eval_single import linear_eval

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Train Contrast on a single GPU')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--data_root', help='path to dataset', type=str, default='./data/')
parser.add_argument('--arch', default='resnet18')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='lr: 0.1 for batch 256')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay for training')
parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--is_knn', action='store_true', help='use knn evalaute')
parser.add_argument('--is_eval', action='store_false', help='use linear evaluate')

# for methods
parser.add_argument('--augs', default='ma', type=str)
parser.add_argument('--method', default='rsa', type=str)
parser.add_argument('--beta', default=0.4, type=float, help='for specific methods')


# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))

    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = args.lr
        else:
            param_group['lr'] = lr


def getAugs(dataset='cifar', augType='aa'):
    if dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 64
    elif dataset == 'tinyimagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        size = 64
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size = 32

    normalize = transforms.Normalize(mean=mean, std=std)

    aggressive_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        normalize
    ])

    weakly_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize
    ])

    frist_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1)),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    second_transform = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if augType == "aw":
        print("Using aggressive-weekly augs")
        train_transform = TwoCropsTransform(aggressive_transform, weakly_transform)
    elif augType == "ma":
        print("Using Multi-stage Augs")
        train_transform = MultiStageTransform(frist_transform, second_transform, normalize)
    else:
        print("Using aggressive-aggressive augs")
        train_transform = TwoCropsTransform(aggressive_transform, aggressive_transform)

    return train_transform, test_transform


def main():
    args = parser.parse_args()
    print(args)
    args.save_freq = 200
    args.eval_epochs = 100
    os.system('nvidia-smi')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    args.logs_dir = 'logs/'
    if not os.path.exists(args.logs_dir):
        os.system('mkdir -p %s' % (args.logs_dir))

    args.model_dir = 'model/'
    if not os.path.exists(args.model_dir):
        os.system('mkdir -p %s' % (args.model_dir))

    # data prepare
    train_transform, test_transform = getAugs(args.dataset, args.augs)
    if args.dataset == 'cifar10':
        train_data = CIFAR10(root=args.data_root, train=True, transform=train_transform, download=True)
        memory_data = CIFAR10(root=args.data_root, train=True, transform=test_transform, download=True)
        test_data = CIFAR10(root=args.data_root, train=False, transform=test_transform, download=True)
    elif args.dataset == 'cifar100':
        train_data = CIFAR100(root=args.data_root, train=True, transform=train_transform, download=True)
        memory_data = CIFAR100(root=args.data_root, train=True, transform=test_transform, download=True)
        test_data = CIFAR100(root=args.data_root, train=False, transform=test_transform, download=True)
    elif args.dataset == 'stl10':
        train_data = STL10(root=args.data_root, split='train+unlabeled', transform=train_transform, download=True)
        memory_data = STL10(root=args.data_root, split='train', transform=test_transform, download=True)
        test_data = STL10(root=args.data_root, split='test', transform=test_transform, download=True)
    elif args.dataset == 'tinyimagenet':
        args.data_root = os.path.join(args.data_root, args.dataset)
        traindir = os.path.join(args.data_root, args.dataset + "-train.lmdb")
        valdir = os.path.join(args.data_root, args.dataset + "-val.lmdb")
        train_data = LmdbDataset(traindir, train_transform)
        memory_data = LmdbDataset(traindir, test_transform)
        test_data = LmdbDataset(valdir, test_transform)

    # create model
    args.max_step = args.epochs * (len(train_data) // args.batch_size)
    model = create_method(args)
    model.cuda()

    optim_params = model.parameters()

    args.lr = args.lr * (args.batch_size // 256)
    optimizer = torch.optim.SGD(optim_params, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    best_acc = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        contrastive_train(model, train_loader, optimizer, epoch, args)

        if args.is_knn:
            val_acc = knn_validate(model.getEncoder(), memory_loader, test_loader, epoch, args)
            best_acc = max(best_acc, val_acc)

        if (epoch + 1) % args.save_freq == 0:
            savename = "model/Train_" + args.dataset + "_" + args.method + "_" + args.augs + "_" + str(epoch + 1) + ".pth.tar"
            torch.save(model.state_dict(), savename)

    if args.is_eval:
        print(getTime(), "Begin to linear evaluate...")
        best_acc = linear_eval(args, model)


if __name__ == '__main__':
    main()
