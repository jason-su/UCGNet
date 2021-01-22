#coding:utf-8
import numpy as np
import os
import cv2
from Constants import baseDir,erzhimap_Dir,raw_val_img_dir,visual_Dir
import scipy.misc as misc


def dot_Visualization(img_data, lbl_data,box_data, save_path, idx):  # 可视化像素点
    w, h = img_data.shape[0], img_data.shape[1]
    image = np.zeros((w, h), np.uint8)
    for xy in lbl_data:
        cv2.circle(img_data, (xy[0],xy[1]), 2, (255,255,255), -1)
#     for b in box_data:
#         cv2.rectangle(img_data, (b[0],b[1]), (b[2],b[3]), (0,255,0), 1)
    cv2.imwrite(save_path + idx, img_data)


if __name__ =="__main__":
#     imgDir = "/home/jjliao/Visdrone_yolo_cluster/VisDrone2019-DET-train/images/"
    imgDir = "/home/jjliao/Visdrone_coco/images/val/"
    txtDir = "/data/data/cluster-detector/erzhimap-yolov4/"
    boxDir = "/data/data/cluster-detector/erzhimap-box/"
    
    images = [i for i in os.listdir(imgDir) if '.jpg' in i]
    labels = [i for i in os.listdir(txtDir) if '.txt' in i]
    print('find image', len(images))
    print('find label', len(labels))

    
    width, height = 600, 600
    for idx,lbl in enumerate(labels):
        if idx>50:
            break
        
        img_id = lbl[:-4]
        img = lbl.replace('txt', 'jpg')
#         img_data = misc.imread(os.path.join(imgDir, img))
        
        imgpath = os.path.join(imgDir, img)
        img_data = cv2.imread(imgpath, -1)  
        print("img path:",imgpath)
        height, width = img_data.shape[:2]  
  
        # 缩小图像  
        size = (int(width*0.25), int(height*0.25))  
#         img_data = cv2.resize(img_data, size, interpolation=cv2.INTER_AREA)  
    
        
        lbl_path = os.path.join(txtDir, lbl)
        box_path = os.path.join(boxDir, lbl)
        
        lbl_data = np.loadtxt(lbl_path,dtype=np.int32,delimiter=",")
#         box_data = np.loadtxt(box_path,dtype=np.int32,delimiter=",")
        if len(lbl_data)==0:
            print("ERROR: empty data:",lbl)
            continue

        dot_Visualization(img_data, lbl_data,[],visual_Dir, img)
        