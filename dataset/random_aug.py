from random import random
from cv2 import imread, imwrite, resize
from math import floor, ceil
import numpy as np
import os

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
        dirty_circle:
        random_flip_horizontal:
    """
    assert random_noise is None, "random_noise is useless"
    assert random_flip_horizontal is None, "random flip horizontal works very badly"
    assert random_filp_vertical is None, "random flip vertical works very badly"
    def random_augmentation_func(ds):
        """
        Params:
            ds: list of [imgs, labels]
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
        return [np.array(tmp_img), np.array(tmp_lab)]
    return random_augmentation_func

random_aug_func = random_augmentation(scale=(0.3, 8),
                                      dirty_circle=0.2,
                                      random_flip_horizontal=None,
                                      random_filp_vertical=None,
                                      random_landmark_mask=0.2,
                                      random_squeeze=[0.3, 6],
                                      random_rotate=0.7)

def read_save_random_aug(img_path, label, prefix, destDir=r'./data/'):
    """Save all data used for train to ./data/
    Params:
        img: image path, string
        label: landmark, list
        postfix: used for avoding duplicate name
    Return:
        new image path
        new label
    """
    img = imread(img_path)
    s = img.shape
    label = np.array(label, dtype=np.float32)
    img = img.astype(np.float32)
    img = img / 255.0
    new_img, label = random_aug_func(img, label)
    new_img = new_img * 255
    new_img = new_img.astype(np.uint8)
    label[::2] = 1.0 * label[::2] / s[1]
    label[1::2] = 1.0 * label[1::2] / s[0]
    new_img_path = os.path.join(os.path.abspath(destDir),
                                str(prefix) + os.path.basename(img_path))
    try:
        imwrite(new_img_path, new_img)
        return new_img_path, label.astype(np.float32)
    except:
        return 0, 0

img_f = open("img_list.txt", 'r')
ldm_f = open("landmark_list.txt", 'r')

for tmp_img_p, tmp_ldm in zip(img_f.readlines(), ldm_f.readlines()):
    if len(tmp_ldm) < 2:
        continue
    else:
        try:
            tmp_img = cv2.imread(tmp_img_p)
        except:
            continue
        if tmp_img:
            
