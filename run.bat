cd dataset
python dataset.py
python dataset.py finetune
cd ..
cd face_detection
python face_detection.py
python face_detection.py finetune
cd ..
cd train
python prepare_dataset.py 40
D:\caffe-windows\scripts\build\tools\Release\caffe.exe train --solver=./solver.prototxt  --gpu=0
cd ..
cd finetune
python finetune_prepare_dataset.py 40
D:\caffe-windows\scripts\build\tools\Release\caffe.exe train --solver=./solver.prototxt  --gpu=0 --weights ..\train\model\_iter_100000.caffemodel
