## FCOS_DET_MASK: Use FCOS model to check whether people wear masks.

## install env
conda create -n fcos python==3.7.   
pip install torch torchvision.   
pip install opencv-python.   
pip install TensorBoard.   

## data
use VOC format, and data can download in https://pan.baidu.com/s/1hssdO_I7vFnSyIw6GuzUBw  Password: 6yr1. 
## parameters change
can modify in model/config.py.     
the backbone support choice in resnet18 and vovnet39.   
## train 
python train.py.  
Focal loss for class:  
<div align=center><img src="https://github.com/2anchao/FCOS_DET_MASK/blob/master/show/focal_loss.png" width="400" height="200" /></div>
Giou loss for reg:  
<div align=center><img src="https://github.com/2anchao/FCOS_DET_MASK/blob/master/show/Giou Loss.png" width="400" height="200" /></div>
cnt loss for better box:  
<div align=center><img src="https://github.com/2anchao/FCOS_DET_MASK/blob/master/show/cntLoss.png" width="400" height="200" /></div>

## inference 
people who wear mask will be mark face_mask label.   
<div align=center><img src="https://github.com/2anchao/FCOS_DET_MASK/blob/master/show/img2.jpg" width="300" height="450" /></div>
people who not wear mask will be mask face label only.   
<div align=center><img src="https://github.com/2anchao/FCOS_DET_MASK/blob/master/show/7c24ab1d2ddbdf65.jpg" width="400" height="300" /></div>

## Attention
The project was completed by me independently for academic exchange. For commercial use, please contact me by email an_chao1994@163.com.  
the code refer ro:https://github.com/VectXmy/FCOS.Pytorch. 

## Other Test
I find use GN in head is better.  
I find use deformable_conv in head is better, I only instead of the first general_conv in head.  refer ro:https://github.com/ChunhuanLin/deform_conv_pytorch.  
I find use ELU in head instead of Relu is better.    
- Please show love to me who is live in the poor mountainous area(optional)
<div align=center><img src="https://github.com/2anchao/FCOS_DET_MASK/blob/master/show/pay.jpeg" width="400" height="300" /></div>