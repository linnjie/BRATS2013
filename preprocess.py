import random

import numpy as np


class CurriculumWrapper:
    def __init__(self, trans, prob):  # if prob, perform trans, otherwise return original
        self.trans = trans  # transformer
        self.prob = prob

    def __call__(self, *args):
        if random.random() < self.prob:
            return self.trans(*args)
        else:
            if len(args) == 1:  # [volume, label]
                args = args[0]
            return args  # volume




class ReColor:  # randomly amplify each channel
    def __init__(self, alpha=0.05):
        self._alpha = alpha

    def __call__(self, img):
        num_chns = img.shape[3]  # HWDC
        t = np.random.uniform(-1, 1, num_chns)
        img = img.astype(np.float32)
        img *= (1 + t * self._alpha)
        return img



class RandomRotate:
    def __init__(self, random_flip=True):
        self.random_flip = random_flip

    def __call__(self, img, mask):  # volume, label
        # horizontal flip
        if self.random_flip and random.random() > 0.5:
            img = img[:, ::-1, :]
            mask = mask[:, ::-1]

        # rotate
        num_rotate = random.randint(0, 3)  # [0, 3]
        if num_rotate > 0:
            img = np.rot90(img, num_rotate)
            mask = np.rot90(mask, num_rotate)
        return img.copy(), mask.copy()  # why copy?

class SampleVolume: # what??
    def __init__(self, dst_shape=[96, 96, 5], pos_ratio=-1):
        self.dst_shape = dst_shape
        self.pos_ratio = pos_ratio

    def __call__(self, data, label):
        src_h, src_w, src_d, _ = data.shape  # HWDC
        dst_h, dst_w, dst_d = self.dst_shape
        if type(dst_d) is list:
            dst_d = random.choice(dst_d)
        if self.pos_ratio < 0:
            h = random.randint(0, src_h - dst_h)
            w = random.randint(0, src_w - dst_w)
            d = random.randint(0, src_d - dst_d)
        else:
            select = label > 0 if random.random() < self.pos_ratio else label == 0
            h, w, d = np.where(select)  # position that meets condition
            select_idx = random.randint(0, len(h) - 1)  # inclusive in both direction
            h = h[select_idx] - int(dst_h / 2)
            w = w[select_idx] - int(dst_w / 2)
            d = d[select_idx] - int(dst_d / 2)
            h = min(max(h, 0), src_h - dst_h)
            w = min(max(w, 0), src_w - dst_w)
            d = min(max(d, 0), src_d - dst_d)
        sub_volume = data[h:h + dst_h, w:w + dst_w, d:d + dst_d, :]
        sub_label = label[h:h + dst_h, w:w + dst_w, d:d + dst_d]
        return sub_volume, sub_label


