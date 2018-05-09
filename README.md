# face landmark caffe implementation


**note** If you are using windows, just run run.bat.

## prepare dataset for train and valid
### define some parameters
```shell
declare WIDTH='64'
```

### generate img_list and landmark list in .txt file
```shell
cd dataset
python dataset.py
python dataset.py finetune
```
This will generate 
    * ./dataset/img_list.txt, save image path for train and test
    * ./dataset/img_list_val.txt, save image path for validation
    * ./dataset/landmark_list.txt, save 25 points x y coordinations of images in img_list.txt
    * ./dataset/img_list_finetune.txt, save image path for later finetune
    * ./dataset/landmark_listffinetune.txt, save 25 points x y coordinations of images in img_list_finetune.txt

### detection
Then we extract face from image, this may take some time(10 mins+) to run.
```shell
cd ..
cd face_detection
python face_detection.py
python face_detection.py finetune
```
In the company(ShangHaiHongMu), I use visual studio 2013 and another detector 
to do the same work in c++ and caffe.

**Note**: To do some test, run `python test_dataset.py`, and some result will show in *./test/*, 
but you need ctrl+c to interrupt this unless you want to test all the faces.

## train a caffe model
Then we need to train a model using caffe.
### 1. First, go to the train dir.
```shell
cd ..
cd train
```
### 2. Prepare the dataset to hdf5
The scripts below will generate
* ./train_dataset.h5
* ./train_dataset.txt
* ./test_dataset.h5
* ./test_dataset.txt

This realy takes time.
```shell
python prepare_dataset.py ${WIDTH}
```

### 3. Train
After generate h5 dataset, we can train the model.
```shell
D:\caffe-windows\scripts\build\tools\Release\caffe.exe train --solver=./solver.prototxt  --gpu=0
```

### 4. Finetune
First generate h5 file.
```shell
cd ..
cd finetune
python finetune_prepare_dataset.py ${WIDTH}
```

Then finetune, change the model to what you want to use.
```shell
D:\caffe-windows\scripts\build\tools\Release\caffe.exe train --solver=./solver.prototxt  --gpu=0 --weights ..\train\model\_iter_100000.caffemodel
```

## Run validation



## *RPN* propose key points

## *regression*