import math
import torch
import torch.nn as nn
import torchvision.models as models

from methods.AbstractMethod import AbstractMethod


class BYOL(nn.Module, AbstractMethod):
    """
    re-implements BYOL https://arxiv.org/abs/2006.07733

    Original implementation uses 4096 and output feature 256

    The predictor uses the same architecture as the projector
    """

    def __init__(self, arch, max_step=3906000):
        """ init additional target and predictor networks """
        super(BYOL, self).__init__()
        self.byol_tau = 0.99
        self.num_step = 0
        self.max_step = max_step

        if arch == "resnet50":
            out_dim = 256
            inner_size = 4096
        else:
            out_dim = 128
            inner_size = 1024

        base_encoder = models.__dict__[arch]
        self.encoder_q = base_encoder(num_classes=out_dim, zero_init_residual=True)
        self.encoder_k = base_encoder(num_classes=out_dim, zero_init_residual=True)
        prev_dim = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Identity()
        self.encoder_k.fc = nn.Identity()

        if arch == 'resnet18':
            print("arch", arch)
            self.encoder_q.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.encoder_q.maxpool = nn.Identity()
            self.encoder_k.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.encoder_k.maxpool = nn.Identity()

        self.encoder_q_head = nn.Sequential(
            nn.Linear(prev_dim, inner_size),
            nn.BatchNorm1d(inner_size),
            nn.ReLU(inplace=True),
            nn.Linear(inner_size, out_dim, False),
        )

        self.encoder_k_head = nn.Sequential(
            nn.Linear(prev_dim, inner_size),
            nn.BatchNorm1d(inner_size),
            nn.ReLU(inplace=True),
            nn.Linear(inner_size, out_dim, False),
        )

        self.pred = nn.Sequential(
            nn.Linear(out_dim, inner_size),
            nn.BatchNorm1d(inner_size),
            nn.ReLU(inplace=True),
            nn.Linear(inner_size, out_dim, False),
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.encoder_q_head.parameters(), self.encoder_k_head.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def update_target(self):
        """ update target network with cosine increasing schedule """
        self.num_step += 1
        tau = 1 - (1 - self.byol_tau) * (math.cos(math.pi * self.num_step / self.max_step) + 1) / 2

        """ copy parameters from main network to target """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_k.data * tau + param_q.data * (1.0 - tau))

        for param_q, param_k in zip(self.encoder_q_head.parameters(), self.encoder_k_head.parameters()):
            param_k.data.copy_(param_k.data * tau + param_q.data * (1.0 - tau))

    def byol_loss(self, image1, image2):
        z1 = self.pred(self.encoder_q_head(self.encoder_q(image1)))
        with torch.no_grad():
            z2 = self.encoder_k_head(self.encoder_k(image2)).detach()

        query = nn.functional.normalize(z1, dim=1)
        target = nn.functional.normalize(z2, dim=1)
        loss = 2 - 2 * (query * target).sum(dim=1)
        return loss

    def forward(self, images):
        im1, im2 = images[0], images[1]
        self.update_target()

        loss_one = self.byol_loss(im1, im2)
        loss_two = self.byol_loss(im2, im1)

        loss = loss_one + loss_two
        return loss.mean()

    def getEncoder(self):
        return self.encoder_q

    def getHead(self):
        return self.encoder_q_head
