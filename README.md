# Animation line art colorization based on optical flow method

## Introduction
This is an implementation for "Animation line art colorization based on optical flow method"

## Prerequisits
Linux  
Pytorch  
GPU+CUDA
```
pip install -r requirements.txt
```

## Dataset
The dataset from [AnimeInterp](https://github.com/lisiyao21/AnimeInterp/)

## Train

```
python3 train.py
```

## Test

* Download the pretrained model from [here](https://pan.baidu.com/s/1wd-IWu4EpqUClcFY_9PLgQ).
提取码：g5k2

* Unzip and move the pre-trained file to checkpoints/\*
* set reference image path
* set sketch image path
* set save path
```
python3 predict.py
```


## References
[Sketch](https://github.com/lllyasviel/sketchKeras)  
[Optical flow](https://github.com/splinter21/ResynNet)  
[Unet](https://github.com/xiaopeng-liao/Pytorch-UNet)


