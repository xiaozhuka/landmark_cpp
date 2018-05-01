# face landmark caffe implementation


**note** If you are using windows, all the scripts can run under cmd with
some modification.

## prepare dataset for train and valid

### generate img_list and landmark list in .txt file
```shell
cd dataset
python dataset.py
```

### detection
```shell
// wrong
g++ -o locate_save_img face_loc.cpp
./locate_save_img
```


## train a caffe model


## *RPN* propose key points

## *regression*