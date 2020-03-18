import json
import os
from lxml.etree import Element, SubElement, tostring
from lxml import etree
from labelme import utils
import numpy as np
import argparse
import cv2


def collate_vocdata(json_dir,out_img,out_xml,out_npy):
    names_list = os.listdir(json_dir)
    label_name_to_value = {'_background_': 0, "slider": 1, "fatigue": 2,"laminar": 3}
    i=0
    for name in names_list:
        if name.split(".")[-1] == "json":
            i += 1
            print("第%d个json文件，名字是："%i,name)
            #生成掩码文件
            per_json=os.path.join(json_dir,name)
            data = json.load(open(per_json))
            img = utils.img_b64_to_arr(data["imageData"])
            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
            base = "wear_"+name.split(".")[0] + ".npy"
            save_npy = os.path.join(out_npy,base)
            np.save(save_npy, lbl)
            #生成xml文件
            width,height=img.shape[0],img.shape[1]
            filename= "wear_"+name.split(".")[0] + ".xml"
            filename_noext = "wear_"+name.split(".")[0]
            anno = GEN_Annotations(filename_noext,width,height)
            save_xml = os.path.join(out_xml,filename)
            make_xml(data, anno, save_xml)
        elif name.split(".")[-1] == "jpg":
            #将原图copy一份放在指定目录下
            save_img_path = os.path.join(out_img,"wear_"+name)
            per_img=os.path.join(json_dir,name)#json_dir is same with img_dir
            os.system("cp "+per_img+" "+save_img_path)
    #将数据分为训练集和测试集
    f_train = open("train.txt","w")#将这两个txt最后移动到ImageSet/Main底下
    f_test = open("test.txt","w")
    names = [name.split(".")[0] for name in names_list if name.split(".")[-1] == "json"]
    data_length = len(names)
    print("数据的总长度是:%d"%data_length)
    split_point=int(data_length*0.7)
    train_names = names[:split_point]
    test_names = names[split_point:]
    for trainn in train_names:
        f_train.write("wear_"+trainn+"\n")
    for testn in test_names:
        f_test.write("wear_"+testn+"\n")


def make_xml(data,  anno, save_path):
    for i in range(len(data['shapes'])):
        points_list = [list(np.asarray(data['shapes'][i]['points']).flatten())]
        x = points_list[0][::2]
        y = points_list[0][1::2]
        xmin,ymin,xmax,ymax = int(min(x)),int(min(y)),int(max(x)),int(max(y))
        label_name = data['shapes'][i]['label']
        anno.add_pic_attr(label_name, xmin, ymin, xmax, ymax)
    anno.savefile(save_path)


class GEN_Annotations:
    def __init__(self, filename, width, height):
        self.root = etree.Element("annotation")
        self.width = width
        self.height = height
        child1 = etree.SubElement(self.root, "folder")
        child1.text = "VOC2007"
        child2 = etree.SubElement(self.root, "filename")
        child2.text = filename
        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(width)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "depth")
        channeln.text = str(3)

    def savefile(self,filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')

    def add_pic_attr(self,label,xmin,ymin,xmax,ymax):
        object = etree.SubElement(self.root, "object")
        namen = etree.SubElement(object, "name")
        namen.text = label
        difficult = etree.SubElement(object, "difficult")
        difficult.text = str(0)
        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(xmin)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(ymin)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(xmax)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(ymax)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make VOC weardata')
    parser.add_argument("json_file",type=str, help="input the json files'dir path")
    parser.add_argument("out_npy", type=str, help="input the out npy files'dir path")
    parser.add_argument("out_img", type=str, help="input the out img files'dir path")
    parser.add_argument("out_xml", type=str, help="input the out xml files'dir path")
    args = parser.parse_args()
    json_dir = args.json_file
    out_img = args.out_img
    out_npy = args.out_npy
    out_xml = args.out_xml
    collate_vocdata(json_dir,out_img,out_xml,out_npy)


#json_file=/Users/chaoan/Desktop/school/学位论文/FCOS_Mask/OriData
#out_npy=/Users/chaoan/Desktop/school/学位论文/FCOS_Mask/ParticleData/VOC2007/Segmentation
#out_img=/Users/chaoan/Desktop/school/学位论文/FCOS_Mask/ParticleData/VOC2007/JPEGImages
#out_xml=/Users/chaoan/Desktop/school/学位论文/FCOS_Mask/ParticleData/VOC2007/Annotations

