# face landmark caffe implementation


**note** If you are using windows, all the scripts can run under cmd with
some modification.

## prepare dataset for train and valid
### define some parameters
```shell
declare WIDTH='64'
```

### generate img_list and landmark list in .txt file
```shell
cd dataset
python dataset.py
```
This will generate 
    * ./dataset/img_list.txt, save image path for train and test
    * ./dataset/img_list_val.txt, save image path for validation
    * ./dataset/landmark_list.txt, save 25 points x y coordinations of images in img_list.txt

### detection
Then we extract face from image, this may take some time(10 mins+) to run.
```shell
cd ..
cd face_detection
python face_detection.py
```
In the company(ShangHaiHongMu), I use visual studio 2013 and another detector 
to do the same work in c++ and caffe.

**Note**: To do some test, run `python test_dataset.py`, and some result will show in *./test/*, 
but you need ctrl+c to interrupt this unless you want to test all the faces.

## train a caffe model
Then we need to train a model using caffe.
1. First, go to the train dir.
```shell
cd ..
cd train
```
2. Prepare the dataset to hdf5
The scripts below will generate
* ./train_dataset.h5
* ./train_dataset.txt
* ./test_dataset.h5
* ./test_dataset.txt
This realy takes time.
```shell
python prepare_dataset.py ${WIDTH}
```

3. Train
```shell
D:\caffe-windows\scripts\build\tools\Release\caffe.exe train --solver=./solver.prototxt  --gpu=0
```

## *RPN* propose key points

## *regression*