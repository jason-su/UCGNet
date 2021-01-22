# coding:utf-8
# xml2txt

import xml.etree.ElementTree as ET
import os
import shutil

from Constants import train_name_txt_1000, crop_1000_xml_dir, train_txt_dir_1000, yolo_trainval_txt_1000, crop_1000_img_dir



classes = ['ignored regions','pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor','others']  # ����ȱ�����ƣ�������xml��ע����һ��


# train_file = 'images_train.txt'    # ���ɵ�txt�ļ�
train_file_txt = ''  # �������txt�ļ�������

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
    in_file = open("{}/{}.xml".format(crop_1000_xml_dir,image_id))
    out_file = open("{}/{}.txt".format(train_txt_dir_1000,image_id),"w")

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    if w==0 or h==0:
        print("Err, w=0 or h=0,",image_id)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:  # ����xml�е�ȱ������
            continue
        cls_id = classes.index(cls)
        if cls_id == 0 or cls_id ==11:
            continue
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id - 1) + " " + " ".join([str(a) for a in bb]) + '\n')


image_ids_train = open(train_name_txt_1000).read().strip().split()  # ��ȡxml�ļ�������

for image_id in image_ids_train:
    convert_annotation(image_id)


anns = os.listdir(crop_1000_xml_dir)
for ann in anns:
    ans = ''
   
    if ann[-3:] != 'xml':
        continue
    train_file_txt = train_file_txt + crop_1000_img_dir + ann[:-3] + 'jpg\n'  # ����yolo��ʽ��ͼƬ����

with open(yolo_trainval_txt_1000, 'w') as outfile:
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


