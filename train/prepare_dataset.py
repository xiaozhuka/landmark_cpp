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

def cf2cl(img):
    """Channel first to channel last.
    """
    i = np.ones([img.shape[1], img.shape[2], img.shape[0]])
    assert img.ndim == 3
    for j in range(img.shape[0]):
        i[:, :, j] = img[j, :, :]
    return i

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
train_ratio = 0.9

def random_augmentation(scale=None,
                        dirty_circle=0.5,
                        random_flip_horizontal=0.5,
                        random_filp_vertical=0.5,
                        random_noise=None,
                        random_landmark_mask=0.5,
                        random_rotate=0.5,
                        random_squeeze=[0.5, 6],
                        test=False):
    """Random augmentate dataset wrapper for parameters.
    Params:
        scale: [probability of scale, ratio of scale]
    """
    assert random_noise is None, "Not supported random_noise"
    assert random_flip_horizontal is None, "random_flip_horizontal works badly"
    assert random_filp_vertical is None, "random_filp_vertical works badly"
    # assert scale is None, "Not supported scale"
    def random_augmentation_func(ds):
        """
        Params:
            ds: [imgs, labels]
        """
        assert ds[0].ndim == 3, "Dataset first element must be images, 3 dim ndarray"

        img = ds[0]
        tmp_img = img.copy()
        tmp_lab = ds[1]
        shape_ori = tmp_img.shape

        # random scale
        if scale is not None:
            if random() < scale[0]:
                tmp_zeros = np.zeros(shape_ori)
                shape_aft = [ floor(tmp * (1.0 - 1.0/scale[1])) for tmp in shape_ori[:2] ]
                tmp_img = resize(tmp_img, shape_aft)
                tmp_lab[::2] = tmp_lab[::2] * (1.0 - 1.0 / scale[1])
                tmp_lab[1::2] = tmp_lab[1::2] * (1.0 - 1.0 / scale[1])

                base_x = floor(random() * (shape_ori[0] - shape_aft[0]))
                base_y = floor(random() * (shape_ori[1] - shape_aft[1]))

                tmp_lab[::2] = tmp_lab[::2] + base_y
                tmp_lab[1::2] = tmp_lab[1::2] + base_x

                tmp_zeros[base_x:(shape_aft[0]+base_x), base_y:(shape_aft[1]+base_y)] = tmp_img
                tmp_img = tmp_zeros

        if random_squeeze is not None:
            if random() < random_squeeze[0]:
                tmp_zeros = np.zeros(shape_ori)
                shape_aft = [floor(tmp * (1.0 - 1.0 / random_squeeze[1])) for tmp in shape_ori[:2]]
                if random() > 0.5:
                    dir = 'x'
                    shape_aft[1] = shape_ori[1]
                    tmp_lab[1::2] = tmp_lab[1::2] * (1.0 - 1.0 / random_squeeze[1])
                else:
                    dir = 'y'
                    shape_aft[0] = shape_ori[0]
                    tmp_lab[0::2] = tmp_lab[0::2] * (1.0 - 1.0 / random_squeeze[1])

                tmp_img = resize(tmp_img, shape_aft)
                tmp_zeros[0:shape_aft[0], 0:shape_aft[1]] = tmp_img
                tmp_img = tmp_zeros

        if dirty_circle is not None:
            if random() < dirty_circle:
                # add some mask
                tmp_for_shape_choice = random()
                rand_color = [np.random.randint(10, 245) for _ in range(3)]
                if tmp_for_shape_choice > 0.5:
                    rand_r = random() * min(shape_ori[:2]) / 8.0
                    rand_x = random() * (min(shape_ori[:2]) - rand_r - 1)
                    rand_y = random() * (min(shape_ori[:2]) - rand_r - 1)
                    cv2.circle(tmp_img, (int(rand_x), int(rand_y)), int(rand_r), rand_color, -1)
                else:
                    rand_w = int(random() * (min(shape_ori[:2]) / 4.0))
                    rand_x = int(random() * (min(shape_ori[:2]) - rand_w - 1))
                    rand_y = int(random() * (min(shape_ori[:2]) - rand_w - 1))
                    cv2.rectangle(tmp_img, (rand_x, rand_y), (rand_x+rand_w, rand_y+rand_w), rand_color, -1)

        if random_rotate is not None:
            if random() < random_rotate:
                tmp_img[tmp_img > 1] = 1
                tmp_angle = (random() - 0.5) * 20
                tmp_img = rotate(tmp_img, angle=tmp_angle)
                center = np.array(tmp_img.shape[:2]) / 2 - 0.5
                for tmp_idx in range(int(len(tmp_lab) / 2)):
                    x = tmp_lab[tmp_idx*2] - center[0]
                    y = tmp_lab[tmp_idx*2+1] - center[1]
                    tmp_lab[tmp_idx*2] = x*math.cos(tmp_angle*3.1416/180) + y*math.sin(tmp_angle*3.1416/180) + center[0]
                    tmp_lab[tmp_idx*2+1] = -x*math.sin(tmp_angle*3.1416/180) + y*math.cos(tmp_angle*3.1416/180) + center[1]

        if random_filp_vertical is not None:
            if random() < random_filp_vertical:
                tmp_img = np.flipud(tmp_img) # up down
                tmp_lab[1::2] = shape_ori[0] - tmp_lab[1::2]

        if random_flip_horizontal is not None:
            if random() < random_flip_horizontal:
                tmp_img = np.fliplr(tmp_img)  # up down
                tmp_lab[::2] = shape_ori[1] - tmp_lab[::2]

        if random_landmark_mask is not None:
            if random() < random_landmark_mask:
                tmp_for_shape_choice = random()
                rand_color = [np.random.randint(10, 245) for _ in range(3)]
                rand_landmark = np.random.randint(0, int(len(tmp_lab) / 2))
                if tmp_for_shape_choice > 0.5:
                    rand_r = int(random() * min(shape_ori[:2]) / 12.0)
                    if rand_r == 0:
                        rand_r = 2
                    rand_x = int(tmp_lab[2*rand_landmark])
                    rand_y = int(tmp_lab[2*rand_landmark+1])
                    if not max([rand_x + rand_r, rand_y + rand_r]) > min(shape_ori):
                        cv2.circle(tmp_img, (rand_x, rand_y), rand_r, rand_color, -1)
                else:
                    rand_w = int(random() * (min(shape_ori[:2]) / 6.0))
                    if rand_w == 0:
                        rand_w = 4
                    rand_x = int(tmp_lab[2 * rand_landmark])
                    rand_y = int(tmp_lab[2 * rand_landmark + 1])
                    if not max([rand_x+rand_w, rand_y+rand_w]) > min(shape_ori):
                        cv2.rectangle(tmp_img, (rand_x, rand_y), (rand_x+rand_w, rand_y+rand_w), rand_color, -1)
        # if not test:
        #     tmp_img = (tmp_img - np.mean(tmp_img)) / np.std(tmp_img)
        return [np.array(tmp_img), np.array(tmp_lab)]
    return random_augmentation_func

def run(prefix=''):
    func = random_augmentation(scale=(0.3, 9),
                                dirty_circle=0.2,
                                random_flip_horizontal=None,
                                random_filp_vertical=None,
                                random_landmark_mask=0.2,
                                random_squeeze=[0.3, 7],
                                random_rotate=0.3)
    ds = [[i, l] for i, l in zip(txt2list(r'..\face_detection\img_list.txt'), txt2list(r'..\face_detection\landmark_list.txt'))]
    
    def _aug(ds):
        result = []
        for single_ds in ds:
            if 'ce' in single_ds[0]:
                result += 6*[single_ds]
            elif 'low' in single_ds[0]:
                result += 2*[single_ds]
            elif '21_points_1' in single_ds[0]:
                result += 2*[single_ds]
            elif '21_points_3' in single_ds[0]:
                result += 2*[single_ds]
            else:
                result += [single_ds]

        return result
    # ds = _aug(ds)
    # ds = ds * 4
    img_lists = []
    labels_lists = []
    
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

        for _ in range(2):
            label = np.array(label, dtype=np.float32)
            img_ = img.copy()
            label_ = label.copy()
            img__, label__ = func([img_, label_])

            s = img__.shape
            label__[::2] = label__[::2] / s[1]
            label__[1::2] = label__[1::2] / s[0]

            # rgb -> bgr
            img__ = img__[:, :, (2, 1, 0)]
            img_lists.append((img__ - np.mean(img__, axis=(0, 1))) / np.std(img__, axis=(0, 1)))
            labels_lists.append(label__)

    bound = int(len(img_lists) * train_ratio)
    train_imgs = np.array(img_lists[:bound])
    train_labels = np.array(labels_lists[:bound])

    test_imgs = np.array(img_lists[bound:])
    test_labels = np.array(labels_lists[bound:])

    with h5py.File('./train_dataset%s.h5'%prefix, 'w') as f:
        f.create_dataset('data', data=train_imgs, dtype=np.float32)
        f.create_dataset('landmarks', data=train_labels, dtype=np.float32)

    with open('./train_dataset%s.txt'%prefix, 'w') as f:
        f.write(os.path.abspath('./train_dataset%s.h5'%prefix) + '\n')

    with h5py.File('./test_dataset%s.h5'%prefix, 'w') as f:
        f.create_dataset('data', data=test_imgs, dtype=np.float32)
        f.create_dataset('landmarks', data=test_labels, dtype=np.float32)

    with open('./test_dataset%s.txt'%prefix, 'w') as f:
        f.write(os.path.abspath('./test_dataset%s.h5'%prefix) + '\n')

if __name__ == '__main__':
    if len(sys.argv) == 2:
        run()
    else:
        run(sys.argv[2])