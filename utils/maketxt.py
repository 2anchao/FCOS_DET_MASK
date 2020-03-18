import os
import argparse

parser = argparse.ArgumentParser(description='Openimage inference')
parser.add_argument('--maskimg_dir', type=str, default=None, required=True, help='images directory')
parser.add_argument('--label_dir', type=str, default=None, required=True, help='xmls directory')
parser.add_argument('--savetxt_dir', type=str, default=None, required=True, help='txts save directory')
args = parser.parse_args()

images_dir = args.maskimg_dir
xmls_dir = args.label_dir
save_dir = args.savetxt_dir

# images_names = os.listdir(images_dir)
# for i,name in enumerate(images_names):
#     if os.path.splitext(name)[-1] in [".jpeg",".JPG"]:
#         sct=os.path.join(images_dir,name)
#         dst=os.path.join(images_dir,os.path.splitext(name)[0]+".jpg")
#         os.rename(sct,dst)

images_names = os.listdir(images_dir)
can_uses = []
for i,name in enumerate(images_names):
    if os.path.splitext(name)[-1] == ".jpg":
        basename = os.path.splitext(name)[0]
        xml_path = os.path.join(xmls_dir,basename+".xml")
        if os.path.exists(xml_path):
            can_uses.append(basename)

split_param = 0.7
data_length = len(can_uses)
print("数据集的长度是：%d"%data_length)
pos = int(data_length*split_param)
train_names = can_uses[:pos]
val_names = can_uses[pos:]

def write_txt(name_list,f):
    for name in name_list:
        f.write(name+"\n")

if __name__ == "__main__":
    traintxt = open(save_dir+"/train.txt","w")
    valtxt = open(save_dir+"/val.txt","w")
    write_txt(train_names,traintxt)
    write_txt(val_names,valtxt)
    traintxt.close()
    valtxt.close()
