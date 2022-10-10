"""
Mean shift eval

"""

import argparse
import os
import random
import time

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import torch.nn.functional as F

from common.tools import ProgressMeter, AverageMeter, accuracy, getTime
from common.LmdbDataset import LmdbDataset

import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Evaluation')
    parser.add_argument('--arch', default='resnet50')
    parser.add_argument('--dataset', default='ImageNet', type=str, help='ImageNet, ImageNet-100')
    parser.add_argument('--data_root', help='path to dataset', type=str, default='')
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--save', default='./output', type=str, help='experiment output directory')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')
    args = parser.parse_args()
    print(args)
    os.system('nvidia-smi')

    msf_eval(args)


def get_eval_default_config():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Evaluation')
    parser.add_argument('--arch', default='resnet50')
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--data_root', help='path to dataset', type=str, default='')
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--save', default='./output', type=str, help='experiment output directory')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')

    parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')
    parser.add_argument('--dataset', default='ImageNet', type=str, help='ImageNet, ImageNet-100')
    return parser.parse_args([])


def load_weights(model, pretrained):
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        state_dict = torch.load(pretrained, map_location="cpu")

        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
            elif k.startswith('module.encoder.') and not k.startswith('module.encoder.fc'):
                # remove prefix
                state_dict[k[len("module.encoder."):]] = state_dict[k]
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))
        raise ValueError('checkpoint not found: ' + pretrained)


def get_model(arch, wts_path):
    if 'resnet' in arch:
        model = models.__dict__[arch]()
        model.fc = nn.Sequential()
        load_weights(model, wts_path)
    else:
        raise ValueError('arch not found: ' + arch)

    for p in model.parameters():
        p.requires_grad = False

    return model


def msf_eval(args):
    args.print_freq = 1000 if args.dataset == "ImageNet" else 100

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    train_transform, val_transform = getAugs()
    traindir = os.path.join(args.data_root, args.dataset + "-train.lmdb")
    if os.path.isfile(traindir):
        valdir = os.path.join(args.data_root, args.dataset + "-val.lmdb")
        train_dataset = LmdbDataset(traindir, train_transform)
        val_dataset = LmdbDataset(valdir, val_transform)
        train_val_dataset = LmdbDataset(traindir, val_transform)
    else:
        traindir = os.path.join(args.data_root, 'train')
        valdir = os.path.join(args.data_root, 'val')
        train_dataset = datasets.ImageFolder(traindir, train_transform)
        val_dataset = datasets.ImageFolder(valdir, val_transform)
        train_val_dataset = datasets.ImageFolder(traindir, train_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    backbone = get_model(args.arch, args.pretrained)
    backbone = nn.DataParallel(backbone).cuda()
    backbone.eval()

    cached_feats = args.save + '/' + args.dataset + '_var_mean.pth.tar'
    if not os.path.exists(cached_feats):
        train_feats, _ = get_feats(train_val_loader, backbone, args)
        train_var, train_mean = torch.var_mean(train_feats, dim=0)
        torch.save((train_var, train_mean), cached_feats)
    else:
        train_var, train_mean = torch.load(cached_feats)

    n_classes = 1000 if args.dataset == "ImageNet" else 100
    linear = nn.Sequential(
        Normalize(),
        FullBatchNorm(train_var, train_mean),
        nn.Linear(get_channels(args.arch), n_classes),
    )
    linear = linear.cuda()

    optimizer = torch.optim.SGD(linear.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.1)
    cudnn.benchmark = True

    best_acc1 = 0
    for epoch in range(0, args.epochs):
        # train for one epoch
        train(train_loader, backbone, linear, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, backbone, linear, args)

        # modify lr
        lr_scheduler.step()

        # remember best acc@1 and save checkpoint
        best_acc1 = max(acc1, best_acc1)

    print(getTime(), "MSF Eval Multi:", best_acc1)


class Normalize(nn.Module):
    def forward(self, x):
        return x / x.norm(2, dim=1, keepdim=True)


class FullBatchNorm(nn.Module):
    def __init__(self, var, mean):
        super(FullBatchNorm, self).__init__()
        self.register_buffer('inv_std', (1.0 / torch.sqrt(var + 1e-5)))
        self.register_buffer('mean', mean)

    def forward(self, x):
        return (x - self.mean) * self.inv_std


def get_channels(arch):
    if arch == 'resnet50':
        c = 2048
    elif arch == 'resnet18':
        c = 512
    else:
        raise ValueError('arch not found: ' + arch)
    return c


def train(train_loader, backbone, linear, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    backbone.eval()
    linear.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = backbone(images)
        output = linear(output)
        loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, backbone, linear, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    backbone.eval()
    linear.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = backbone(images)
            output = linear(output)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)


def getAugs(dataset='imagenet'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


def get_feats(loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(len(loader), [batch_time], prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    feats, labels, ptr = None, None, 0

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            cur_targets = target.cpu()
            cur_feats = normalize(model(images)).cpu()
            B, D = cur_feats.shape
            inds = torch.arange(B) + ptr

            if not ptr:
                feats = torch.zeros((len(loader.dataset), D)).float()
                labels = torch.zeros(len(loader.dataset)).long()

            feats.index_copy_(0, inds, cur_feats)
            labels.index_copy_(0, inds, cur_targets)
            ptr += B

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return feats, labels


if __name__ == '__main__':
    main()
