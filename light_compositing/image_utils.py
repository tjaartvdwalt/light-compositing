import numpy as np


class ImageUtils():
    def __init__(self):
        pass


def normalize(img):
    div = np.ones(img.shape, np.float32)
    div = div / 255

    return img * div


def denormalize(img):
    div = img * 255
    return np.uint8(div)
