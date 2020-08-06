import random
import numpy as np
import cv2

from utils.transforms import CustomTransform

class Darkness(CustomTransform):
    def __init__(self, coeff):
        assert coeff >= 1., "Darkness coefficient must be greater than 1"
        self.coeff = coeff

    def __call__(self, sample):
        img = sample.get('img')
        coeff = np.random.uniform(1., self.coeff)
        img = (img.astype('float32') / coeff).astype('uint8')

        _sample = sample.copy()
        _sample['img'] = img
        return _sample