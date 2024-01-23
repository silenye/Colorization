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

## Video
We use the predicted frames as reference frames again for colorization of the line sketch frames. Each video is colored 30 frames. Therefore the quality of the frames is not as high the further you go. In practice, we recommend a more intensive setting of reference frames to colorize, especially for high-speed motion frames.

![image](https://github.com/silenye/Colorization/blob/master/video/116_gif-converter.gif?raw=true)
![image](https://github.com/silenye/Colorization/blob/master/video/103_gif-converter.gif?raw=true)


## References
[Sketch](https://github.com/lllyasviel/sketchKeras)  
[Optical flow](https://github.com/splinter21/ResynNet)  
[Unet](https://github.com/xiaopeng-liao/Pytorch-UNet)


