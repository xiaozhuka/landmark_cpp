# -*- coding: utf-8 -*-

"""Some operations.
"""
import numpy as np
import h5py

from math import floor
import math
from cv2 import imread
import cv2
from skimage.transform import resize, rotate

from random import random
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

def cl2cf(img):
    """Channel last to channel first.
    """
    i = np.ones([img.shape[1], img.shape[2], img.shape[0]])
    assert img.ndim == 3
    for j in range(img.shape[-1]):
        i[j, :, :] = img[:, :, j]
    return i
def txt2list(txt_file):
    """Return list returned by f.readlines()
    """
    with open(txt_file, 'r') as f:
        l = list(f.readlines())
    return [t.strip() for t in list(l)]

width = height = int(sys.argv[1])

def run(prefix=''):
    ds = [[i, l] for i, l in zip(txt2list(r'..\face_detection\img_list_finetune.txt'),
                                 txt2list(r'..\face_detection\landmark_listfinetune.txt'))]

    img_lists = []
    labels_lists = []

    def _aug(ds):
        result = []
        for single_ds in ds:
            if 'ce' in single_ds[0]:
                result += 10*[single_ds]
            elif 'glasses' in single_ds[0]:
                result += 10*[single_ds]
            else:
                result += [single_ds]

        return result
    ds = _aug(ds)

    np.random.shuffle(ds)
    filenames = [ds[i][0] for i in range(len(ds))]
    labels = [[float(t) for t in ds[i][1].split()] for i in range(len(ds))]
    for filename, label in zip(filenames, labels):
        if len(label) < 2:
            continue
        if not os.path.isfile(filename):
            print('filename: %s don\' exists' % filename)
            continue
        img = imread(filename)
        if img.shape[0] < height or img.shape[1] < width:
            continue
        if img is None:
            print('Failed to read file: %s'%filename)
            continue
        print('Read in image: %s'%filename)
        img = resize(img, (width, height))
        label = np.array(label, dtype=np.float32)
        img = img.astype(np.float32)
        img = img / 255.0

        s = img.shape
        label[::2] = label[::2] / s[1]
        label[1::2] = label[1::2] / s[0]
        img = (img - np.mean(img, axis=(0, 1))) / np.std(img, axis=(0, 1))
        img = cl2cf(img)
        img_lists.append(img)
        # img_lists.append((img[:, :, ::-1] - np.mean(img)) / np.std(img))
        labels_lists.append(label)

        # for _ in range(1):
        #     label = np.array(label, dtype=np.float32)
        #     img_ = img.copy()
        #     label_ = label.copy()
        #     img__, label__ = func([img_, label_])

        #     s = img__.shape
        #     label__[::2] = label__[::2] / s[1]
        #     label__[1::2] = label__[1::2] / s[0]

        #     # rgb -> bgr
        #     img__ = img__[:, :, (2, 1, 0)]
        #     img_lists.append((img__ - np.mean(img__, axis=(0, 1))) / np.std(img__, axis=(0, 1)))
        #     labels_lists.append(label__)

    train_imgs = np.array(img_lists)
    train_labels = np.array(labels_lists)
    with h5py.File('./train_dataset%s.h5'%prefix, 'w') as f:
        f.create_dataset('data', data=train_imgs, dtype=np.float32)
        f.create_dataset('landmarks', data=train_labels, dtype=np.float32)

    with open('./train_dataset%s.txt'%prefix, 'w') as f:
        f.write(os.path.abspath('./train_dataset%s.h5'%prefix) + '\n')

if __name__ == '__main__':
    run()