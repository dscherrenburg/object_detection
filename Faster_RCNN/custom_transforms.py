import torch
from torchvision.transforms import functional as F

class Resize(object):
    def __init__(self, size: int):
        self.size = size  # Target size for the longest side

    def __call__(self, image, target):
        h, w = image.shape[-2:]  # Get height and width
        scale = self.size / max(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        # Resize image and target boxes
        image = F.resize(image, [new_h, new_w])
        return image, target
        

class ToTensor(object):
    def __call__(self, image, target):
        image = torch.Tensor(image)
        return image, target


class CustomTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)  # Apply each transform to both image and target
        return image, target