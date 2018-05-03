"""This file will generate two .txt file one for
validation, the other train and test with each line
represente a image path.

The image endswith {1}_0.jpg
The landmark {1}.txt

Generate:
    img_list_val.txt
    img_list.txt
    landmark_list_val.txt
"""

import glob
import os
import cv2

train_ratio = 0.9
test_ratio = 1 - train_ratio

def generate_file_list_txt(target_txt, parent_path, glob_par="*.jpg"):
    """This function is specified for my project, it
    will generate a txt file with every single line for
    one image file path.
    Params:
        target_txt: txt file name
        parent_path: parent path for images
    """
    with open(target_txt, 'w') as f:
        for single_parent_path in parent_path:
            glob_res = glob.glob(os.path.join(single_parent_path, glob_par))
            glob_res = filter(lambda x: not os.path.basename(x).startswith('dlib_'),
                             glob_res)
            f.write('\n'.join(glob_res))
            f.write('\n')

def generate_file_list_txt_l(target_txt, parent_path, glob_par="*.jpg"):
    """This function is specified for my project, it
    will generate a txt file with every single line for
    one image file path.
    Params:
        target_txt: txt file name
        parent_path: parent path for images
    """
    with open(target_txt, 'w') as f:
        for single_parent_path in parent_path:
            glob_res = []
            dirs = os.listdir(single_parent_path)
            for tmp_path in dirs:
                if tmp_path.endswith(glob_par[1:]):
                    glob_res.append(os.path.join(single_parent_path, tmp_path))
            glob_res = filter(lambda x: not os.path.basename(x).startswith('dlib_'),
                             glob_res)
            f.write('\n'.join(glob_res))
            f.write('\n')

def save_landmark(imgTxt=r'img_list.txt'):
    """Save landmark too.
    Generate landmark_list_val.txt
    """
    with open(imgTxt, 'r') as img_f:
        with open('landmark_'+''.join(imgTxt.split('_')[1:]), 'w') as ldm_f:
            for img_path in img_f.readlines():
                ldm_path = os.path.join(os.path.dirname(img_path),
                                        os.path.basename(img_path)[:-7] + '.txt')
                if os.path.isfile(ldm_path):
                    with open(ldm_path, 'r') as tmp_f:
                        tmp_lab = ''
                        for tmp_line in tmp_f.readlines():
                            if tmp_line.startswith('sensetime_21_points'):
                                tmp_lab += (' '.join(tmp_line.split()[1:]))
                                continue
                            if tmp_line.startswith('sensetime_106_points'):
                                tmp_list = tmp_line.split()
                                tmp_lab += " "
                                tmp_lab += (' '.join(tmp_list[1:3] + tmp_list[5:7]))
                                tmp_lab += ' '
                                tmp_lab += (' '.join(tmp_list[61:63] + tmp_list[65:67]))
                                break
                    ldm_f.write(tmp_lab + '\n')
                else:
                    ldm_f.write('\n')

generate_file_list_txt('img_list_val.txt',
                        [r'E:\python_vanilla\validation_dataset\ce',
                        r'E:\python_vanilla\validation_dataset\di',
                        r'E:\python_vanilla\validation_dataset\glasses',
                        r'E:\python_vanilla\validation_dataset\hat',
                        r'E:\python_vanilla\validation_dataset\tai',
                        r'E:\python_vanilla\validation_dataset\zheng']
)

generate_file_list_txt('img_list.txt',
                        [r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\25points_selected',
                        r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\ce',
                        r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\hu',
                        r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\low',
                        r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\21_points_1',
                        r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\21_points_2',
                        r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\21_points_3'],
                        '*_0.jpg'
                        )

save_landmark('img_list.txt')