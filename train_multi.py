#!/usr/bin/env python

import argparse
import builtins
import math
import os
import random
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.cuda.amp import autocast

from methods import create_method
from common.tools import getTime, AverageMeter, ProgressMeter
from common.augments import TwoCropsTransform, MultiStageTransform, GaussianBlur
from common.LmdbDataset import LmdbDataset
from common.lars import LARS
from eval_multi import get_eval_default_config, msf_eval

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Train Contrast on multiple GPUs')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--dataset', default='ImageNet', type=str, help='ImageNet, ImageNet-100')
parser.add_argument('--data_root', help='path to dataset', type=str, default='')
parser.add_argument('--arch', default='resnet50')
parser.add_argument('--lr', '--learning-rate', default=0.2, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--amp', action='store_true', help='amp, normal')
parser.add_argument('--is_eval', action='store_false', help='use linear evaluate')

# for methods
parser.add_argument('--method', default='rsa', type=str)
parser.add_argument('--augs', default='ma', type=str)
parser.add_argument('--warmup-epochs', default=0, type=int, metavar='N', help='number of warmup epochs')
parser.add_argument('--beta', default=0.4, type=float, help='for specific methods')


def getAugs(dataset='imagenet', augType='aa'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    aggressive_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        normalize
    ])

    weakly_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    frist_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
    ])

    second_transform = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    ])

    if augType == "aw":
        print("Using aggressive-weekly augs")
        train_transform = TwoCropsTransform(aggressive_transform, weakly_transform)
    elif augType == "ma":
        print("Using Multi-stage augs")
        train_transform = MultiStageTransform(frist_transform, second_transform, normalize)
    else:
        train_transform = TwoCropsTransform(aggressive_transform, aggressive_transform)

    return train_transform


def main():
    print("Train with Multiple GPUs")
    args = parser.parse_args()
    print(args)
    args.distributed = True
    os.system('nvidia-smi')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.logs_dir = 'logs/'
    if not os.path.exists(args.logs_dir):
        os.system('mkdir -p %s' % (args.logs_dir))

    args.model_dir = 'model/'
    if not os.path.exists(args.model_dir):
        os.system('mkdir -p %s' % (args.model_dir))

    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.rank = gpu

    # suppress printing if not master
    if args.gpu != 0:
        def print_pass(*args, flush=False):
            pass
        builtins.print = print_pass

    dist.init_process_group(backend="nccl", init_method="tcp://localhost:10001", world_size=ngpus_per_node, rank=args.rank)

    # Data loading code
    train_transform = getAugs(args.dataset, args.augs)
    traindir = os.path.join(args.data_root, args.dataset + "-train.lmdb")
    if os.path.isfile(traindir):
        train_dataset = LmdbDataset(traindir, train_transform)
    else:
        traindir = os.path.join(args.data_root, 'train')
        train_dataset = datasets.ImageFolder(traindir, train_transform)

    args.lr = args.lr * (args.batch_size // 256)
    args.print_freq = len(train_dataset) // args.batch_size // 5
    args.max_step = args.epochs * (len(train_dataset) // args.batch_size)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = create_method(args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) if args.dataset == "ImageNet" else model
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    optim_params = model.parameters()

    if args.dataset == "ImageNet-100":
        optimizer = torch.optim.SGD(optim_params, args.lr, momentum=0.9, weight_decay=args.wd)
    else:
        optim_params = add_weight_decay(model, args.wd)
        optimizer = LARS(optim_params, args.lr, weight_decay=0, momentum=0.9)
    cudnn.benchmark = True

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    args.batch_size = int(args.batch_size / ngpus_per_node)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=4, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train(train_loader, model, optimizer, epoch, args)

    if args.rank % ngpus_per_node == 0:
        savename = "model/TrainMulti_" + args.dataset + "_" + args.method + "_" + str(epoch + 1) + "_beta" + str(args.beta) + "_lr" + str(args.lr) + "_wd" + str(args.wd) + ".pth.tar"
        torch.save(model.state_dict(), savename)
        del model

        if args.is_eval:
            print(getTime(), "Begin to linear evaluate...")
            eval_args = get_eval_default_config()
            eval_args.pretrained = savename
            eval_args.dataset = args.dataset
            eval_args.data_root = args.data_root
            msf_eval(eval_args)
    else:
        del model


def train(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses], prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()

    iter_num = len(train_loader)
    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        adjust_learning_rate(optimizer, epoch + i / iter_num, args)

        for j in range(len(images)):
            images[j] = images[j].cuda(args.gpu, non_blocking=True)

        if args.amp:
            with autocast():
                results = model(images)
                if isinstance(results, tuple) and len(results) == 3:
                    loss, loss_w, loss_s = results
                else:
                    loss = results

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            results = model(images)
            if isinstance(results, tuple) and len(results) == 3:
                loss, loss_w, loss_s = results
            else:
                loss = results

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        losses.update(loss.item(), images[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or (i + 1) == iter_num:
            progress.display(i)


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))

    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = args.lr
        else:
            param_group['lr'] = lr
    return lr


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """Splits param group into weight_decay / non-weight decay.
       Tweaked from https://bit.ly/3dzyqod
    :param model: the torch.nn model
    :param weight_decay: weight decay term
    :param skip_list: extra modules (besides BN/bias) to skip
    :returns: split param group into weight_decay/not-weight decay
    :rtype: list(dict)
    """
    # if weight_decay == 0:
    #     return model.parameters()

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0},
            {'params': decay, 'weight_decay': weight_decay}]


if __name__ == '__main__':
    main()
