"""
Using pred representations calculates losses.

"""

import math
import torch
import torch.nn as nn
from methods.BYOL import BYOL


class RSA(BYOL):

    def __init__(self, arch='resnet18', max_step=3906000, beta=1.):
        super().__init__(arch, max_step)
        self.beta = beta

    def rsa_loss(self, im1_w, im2_w, im1_a, im2_a):
        q1_a = self.pred(self.encoder_q_head(self.encoder_q(im1_a)))
        q2_a = self.pred(self.encoder_q_head(self.encoder_q(im2_a)))
        q1_a = nn.functional.normalize(q1_a, dim=1)
        q2_a = nn.functional.normalize(q2_a, dim=1)

        with torch.no_grad():
            z1_w = self.encoder_k_head(self.encoder_k(im1_w)).detach()
            z2_w = self.encoder_k_head(self.encoder_k(im2_w)).detach()
            z1_w = nn.functional.normalize(z1_w, dim=1)
            z2_w = nn.functional.normalize(z2_w, dim=1)

        a1w2_loss = self.mse_loss(q1_a, z2_w)
        a1a2_loss = self.mse_loss(q1_a, q2_a.clone().detach())
        a2w1_loss = self.mse_loss(q2_a, z1_w)
        a2a1_loss = self.mse_loss(q2_a, q1_a.clone().detach())

        beta = self.beta * (math.cos(math.pi * self.num_step / self.max_step) + 1) / 2

        # 1 aggressive-aggressive; 0 aggressive-weak; 0.5 aggressive-aggressive/weak;
        loss1 = beta * a1a2_loss + (1 - beta) * a1w2_loss
        loss2 = beta * a2a1_loss + (1 - beta) * a2w1_loss

        return loss1 + loss2, a1w2_loss + a2w1_loss, a1a2_loss + a2a1_loss

    def forward(self, images):
        self.update_target()

        im1_w, im2_w, im1_a, im2_a = images[0], images[1], images[2], images[3]
        loss, loss_w, loss_a = self.rsa_loss(im1_w, im2_w, im1_a, im2_a)

        return loss.mean(), loss_w.mean(), loss_a.mean()

    def mse_loss(self, feature1, feature2):
        return (2 - 2 * (feature1 * feature2).sum(dim=1))
