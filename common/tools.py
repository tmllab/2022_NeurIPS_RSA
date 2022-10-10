import time
import datetime
import shutil

import torch
import torch.nn.functional as F


def getTime():
    time_stamp = datetime.datetime.now()
    return time_stamp.strftime('%H:%M:%S')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch + 1)]
        entries += [str(meter) for meter in self.meters]
        # not cache
        print(getTime(), '\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, filename, is_best=False):
    savename = "model/" + filename + ".pth.tar"
    best_filename = "model/" + filename + "_best.pth.tar"

    torch.save(state, savename)
    if is_best:
        shutil.copyfile(savename, best_filename)


# train for one epoch
def contrastive_train(net, data_loader, train_optimizer, epoch, args, supervised=False):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_w = AverageMeter('Loss_w', ':.4e')
    losses_s = AverageMeter('Loss_s', ':.4e')
    progress = ProgressMeter(len(data_loader), [batch_time, data_time, losses, losses_w, losses_s], prefix="Epoch: [{}]".format(epoch))

    net.train()
    end = time.time()
    for i, (images, labels) in enumerate(data_loader):
        for j in range(len(images)):
            images[j] = images[j].cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        results = net(images)
        if isinstance(results, tuple) and len(results) == 3:
            loss, loss_w, loss_s = results
        else:
            loss = results

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        losses.update(loss.item(), images[0].size(0))
        if isinstance(results, tuple) and len(results) == 3:
            losses_w.update(loss_w.item(), images[0].size(0))
            losses_s.update(loss_s.item(), images[0].size(0))

        batch_time.update(time.time() - end)
        end = time.time()

    progress.display(i)


def knn_validate(net, memory_data_loader, test_data_loader, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(test_data_loader),
        [batch_time, top1],
        prefix='Test: ')

    # switch to evaluate mode
    net.eval()
    feature_bank = []
    classes = len(memory_data_loader.dataset.classes)
    with torch.no_grad():
        end = time.time()

        # generate feature bank
        for data, target in memory_data_loader:
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        if args.dataset == "stl10":
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device, dtype=torch.int64)
        else:
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device, dtype=torch.int64)
        # loop test data to predict the label by weighted knn search

        for i, (images, target) in enumerate(test_data_loader):
            images = images.cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)

            # compute output
            feature = net(images)
            feature = F.normalize(feature, dim=1)
            output = knn_predict(feature, feature_bank, feature_labels, classes, 200, 0.1).cpu()

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))[0]
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        progress.display(i)

    return top1.avg


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    # pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_scores
