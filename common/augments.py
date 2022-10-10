import random
from PIL import ImageFilter, ImageOps
import torchvision.transforms as transforms


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, second_transform=None):
        self.base_transform = base_transform
        self.second_transform = second_transform

    def __call__(self, x):
        q = self.base_transform(x)

        if self.second_transform is None:
            k = self.base_transform(x)
        else:
            k = self.second_transform(x)

        return [q, k]


class MultiStageTransform:

    def __init__(self, frist_transform, second_transform, normalize):
        self.frist_transform = frist_transform
        self.second_transform = second_transform
        self.normalize = normalize

    def __call__(self, x):
        x1_stage1_tmp = self.frist_transform(x)
        x2_stage1_tmp = self.frist_transform(x)

        x1_stage2_tmp = self.second_transform(x1_stage1_tmp)
        x2_stage2_tmp = self.second_transform(x2_stage1_tmp)

        x1_stage1 = self.normalize(transforms.ToTensor()(x1_stage1_tmp))
        x2_stage1 = self.normalize(transforms.ToTensor()(x2_stage1_tmp))
        x1_stage2 = self.normalize(transforms.ToTensor()(x1_stage2_tmp))
        x2_stage2 = self.normalize(transforms.ToTensor()(x2_stage2_tmp))

        return [x1_stage1, x2_stage1, x1_stage2, x2_stage2]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize():
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, sample):
        return ImageOps.solarize(sample, self.threshold)
