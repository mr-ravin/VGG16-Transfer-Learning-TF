# VGG16-Transfer-Learning-TF
This repository contains the transfer learning over VGG16 architecture implemented in Tensorflow, using tf.slim

####  Author: [Ravin Kumar](https://mr-ravin.github.io)
### Libraries required:
- #### Tensorflow version : 1.12
- #### OpenCV version : 2

### Directory Architecture:
```
|
|-transfer_learning.py 
|
|-inference.py 
|
|-vgg_16.ckpt (link is provided to download pretrained weights)
|
|-time (stores log file for overall time, start time, auto-shutdown time)
|
|-saved (trained weights)
|
|-log (stores visual graph for tensorboard)
|
|-data
   |-Class1 (contains *.jpg for class1)
   |
   |-Class2 (contains *.jpg for class2)
   |
   |-Class3 (contains *.jpg for class3)
   |
   |-Class4 (contains *.jpg for class4)
```

#### Training code
```python
python3 transfer_learning.py
```

#### Inference code
```python
python3 inference.py
```

### Download VGG16.ckpt file from Google Drive
[Download](https://drive.google.com/file/d/1M8YIeVplrx1fuPBblEZQmDCOSgRLUABc/view?usp=sharing)



