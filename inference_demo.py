
import cv2
from model.fcos import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
from dataloader.VOC_dataset import VOCDataset
import time
import os
from model.config import DefaultConfig as cfg


if __name__=="__main__":
    model=FCOSDetector(mode="inference",config=cfg)
    model.load_state_dict(torch.load("FCOSMASK_epoch61_loss1.0623.pth",map_location=torch.device('cpu')))
    model=model.cuda().eval()
    print("===>success loading model")
    root=cfg.inference_dir
    names=os.listdir(root)
    for name in names:
        img_pad=cv2.imread(root+name)
        img=img_pad.copy()
        img_t=torch.from_numpy(img).float().permute(2,0,1)
        img1= transforms.Normalize([102.9801, 115.9465, 122.7717],[1.,1.,1.])(img_t)
        img1=img1.cuda()
        
        start_t=time.time()
        with torch.no_grad():
            out=model(img1.unsqueeze_(dim=0))
        end_t=time.time()
        cost_t=1000*(end_t-start_t)
        print("===>success processing img, cost time %.2f ms"%cost_t)
        # print(out)
        scores,classes,boxes=out

        boxes=boxes[0].cpu().numpy().tolist()
        classes=classes[0].cpu().numpy().tolist()
        scores=scores[0].cpu().numpy().tolist()

        for i,box in enumerate(boxes):
            pt1=(int(box[0]),int(box[1]))
            pt2=(int(box[2]),int(box[3]))
            img_pad=cv2.rectangle(img_pad,pt1,pt2,(0,255,0))
            img_pad=cv2.putText(img_pad,"%s %.3f"%(VOCDataset.CLASSES_NAME[int(classes[i])],scores[i]),(int(box[0]),int(box[1])+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,[0,200,20],2)

        save_dir = "out_images/"
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(save_dir+name,img_pad)





