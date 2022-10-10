import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10

from methods import create_method
from common.LmdbDataset import LmdbDataset
from common.tools import ProgressMeter, AverageMeter, accuracy


class LinearHead(nn.Module):
    def __init__(self, net, dim_in=128, num_class=1000):
        super().__init__()
        self.net = net
        self.fc = nn.Linear(dim_in, num_class)

        for param in self.net.parameters():
            param.requires_grad = False

        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        with torch.no_grad():
            feat = self.net(x)
        return self.fc(feat)


def linear_eval(args, pre_train=None):
    if args.dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 64
    elif args.dataset == 'tinyimagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        size = 64
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size = 32

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize])

    test_transform = transforms.Compose([
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize])

    if args.dataset == 'cifar10':
        train_dataset = CIFAR10(root='data', download=True, transform=train_transform)
        test_dataset = CIFAR10(root='data', train=False, download=True, transform=test_transform)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = CIFAR100(root='data', download=True, transform=train_transform)
        test_dataset = CIFAR100(root='data', train=False, download=True, transform=test_transform)
        num_classes = 100
    elif args.dataset == 'stl10':
        train_dataset = STL10(root=args.data_root, split='train', transform=train_transform, download=True)
        test_dataset = STL10(root=args.data_root, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'tinyimagenet':
        traindir = os.path.join(args.data_root, args.dataset + "-train.lmdb")
        valdir = os.path.join(args.data_root, args.dataset + "-val.lmdb")
        train_dataset = LmdbDataset(traindir, train_transform)
        test_dataset = LmdbDataset(valdir, test_transform)
        num_classes = 200

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Load the model if None
    if pre_train is None:
        # create model
        pre_train, supervised = create_method(args)
        state_dict = torch.load(args.pretrained, map_location='cpu')
        pre_train.load_state_dict(state_dict)

    dim_in = 512
    model = LinearHead(pre_train.getEncoder(), dim_in=dim_in, num_class=num_classes).cuda()

    lr = 30
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9, weight_decay=0, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    torch.backends.cudnn.benchmark = True

    best_acc = 0
    for epoch in range(args.eval_epochs):
        # ---------------------- Train --------------------------
        model.train()

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            train_loader.__len__(),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch)
        )

        end = time.time()
        for i, (image, label) in enumerate(train_loader):
            data_time.update(time.time() - end)
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            out = model(image)
            loss = F.cross_entropy(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item())

            if i % 100 == 0:
                progress.display(i)

        scheduler.step()

        # ---------------------- Test --------------------------
        model.eval()
        top1 = AverageMeter('Acc@1', ':6.2f')
        with torch.no_grad():
            end = time.time()
            for i, (image, label) in enumerate(test_loader):
                image = image.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                # compute output
                output = model(image)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, label, topk=(1, 5))
                top1.update(acc1[0], image.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        best_acc = max(top1.avg, best_acc)
        print('Epoch:{} * Acc@1 {top1_acc:.3f} Best_Acc@1 {best_acc:.3f}'.format(epoch, top1_acc=top1.avg, best_acc=best_acc))

    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='resnet18')
    parser.add_argument('--method', type=str, default='')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--eval_epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
    # parser.add_argument('--workers', default=8, type=int, metavar='N')

    args = parser.parse_args()
    print(args)
    model = linear_eval(args)
    torch.save(model.state_dict(), "model/Eval_" + args.dataset + "_" + args.method + ".pth.tar")


if __name__ == "__main__":
    main()
