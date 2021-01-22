# coding:utf-8
# xml2txt

import xml.etree.ElementTree as ET
import os
import shutil

from Constants import train_name_txt,clip_xml_dir,train_txt_dir,yolo_trainval_txt,clip_img_dir


if os.path.exists(train_txt_dir):
    shutil.rmtree(train_txt_dir)#删除再建立
    os.makedirs(train_txt_dir)
classes = ['ignored regions','pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor','others']  # 输入缺陷名称，必须与xml标注名称一致


# train_file = 'images_train.txt'    # 生成的txt文件
train_file_txt = ''  # 存放生成txt文件的内容

# wd = os.getcwd()

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    box = list(box)
    box[1] = min(box[1], size[0])
    box[3] = min(box[3], size[1])
    x = ((box[0] + box[1]) / 2.0) * dw
    y = ((box[2] + box[3]) / 2.0) * dh
    w = (box[1] - box[0]) * dw
    h = (box[3] - box[2]) * dh
    return (x, y, w, h)   
  


def convert_annotation(image_id):
    in_file = open("{}/{}.xml".format(clip_xml_dir,image_id))
    out_file = open("{}/{}.txt".format(train_txt_dir,image_id),"w")
#     in_file = open('/home/jjliao/Visdrone_yolo_cluster/VisDrone2019-DET-train/annotations_cluster_xml/%s.xml' % (image_id))  # 读取xml文件路径
#     out_file = open('/home/jjliao/Visdrone_yolo_cluster/labels/train/%s.txt' % (image_id), 'w')  # 需要保存的txt格式文件路径
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    if w==0 or h==0:
        print("Err, w=0 or h=0,",image_id)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:  # 检索xml中的缺陷名称
            continue
        cls_id = classes.index(cls)
        if cls_id == 0 or cls_id ==11:
            continue
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id - 1) + " " + " ".join([str(a) for a in bb]) + '\n')


# image_ids_train = open('/home/jjliao/Visdrone_yolo_cluster/train_name.txt').read().strip().split()  # 读取xml文件名索引
image_ids_train = open(train_name_txt).read().strip().split()  # 读取xml文件名索引

for image_id in image_ids_train:
    convert_annotation(image_id)

# anns = os.listdir('./VisDrone2019-DET-train/annotations_cluster_xml/')  # xml标注文件的目录
anns = os.listdir(clip_xml_dir)
for ann in anns:
    ans = ''
#     outpath = yolo_trainval_dir + '/labels/train/' + ann  
   
    if ann[-3:] != 'xml':
        continue
    train_file_txt = train_file_txt + clip_img_dir + ann[:-3] + 'jpg\n'  # 保存yolo格式的图片索引

with open(yolo_trainval_txt, 'w') as outfile:
    outfile.write(train_file_txt)

# # 复制visdrone_train里图片复制到images中
# image_old = "/home/jjliao/Visdrone_yolo_cluster/VisDrone2019-DET-train/images_cluster/"
# image_new = yolo_trainval_dir + "/images/train/"
# 
# if not os.path.exists(image_new):
#     os.makedirs(image_new)
# for file in os.listdir(image_old):
#     full_file = os.path.join(image_old, file)
#     new_full_file = os.path.join(image_new, file)
#     shutil.copy(full_file, new_full_file)

