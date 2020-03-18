## FCOS_DET_MASK: Use FCOS model to check whether people wear masks.

## install env
conda create -n fcos python==3.7.   
pip install torch torchvision.   
pip install opencv-python.   
pip install TensorBoard.   

## data
use VOC format, and data can download in https://pan.baidu.com/s/1hssdO_I7vFnSyIw6GuzUBw  ->6yr1. 
## parameters change
can modify in model/config.py.   

## train 
python train.py.  

## inference 
people who wear mask will be mark face_mask label.   
<div align=center><img src="https://github.com/2anchao/FCOS_DET_MASK/blob/master/show/img2.jpg" width="400" height="500" /></div>
people who not wear mask will be mask face label only.   
<div align=center><img src="https://github.com/2anchao/FCOS_DET_MASK/blob/master/show/7c24ab1d2ddbdf65.jpg" width="500" height="360" /></div>
