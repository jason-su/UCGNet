#coding:utf-8
import numpy as np
import os
import cv2
from Constants import baseDir,erzhimap_Dir,raw_val_img_dir,visual_Dir
import scipy.misc as misc


def dot_Visualization(img_data, box_data,save_path, idx):  # 可视化像素点
    w, h = img_data.shape[0], img_data.shape[1]
    for b in box_data:
        if float(b[4])>=0.5:
            cv2.rectangle(img_data, (b[0],b[1]), (b[2],b[3]), (0,255,0), 1)
    cv2.imwrite(save_path + idx, img_data)

def readMergeResult(txtFile):
    data = {}
    f = open(txtFile,"r")
    for line in f.readlines():
        arr = np.array(line.strip('\n').split(","))
        k_id = ("{}".format(arr[0]))
        if not k_id in data.keys():
            data[k_id] = []
        n_arr = np.array(arr[2:7],dtype=np.float32)
        n_arr[3] += n_arr[1]
        n_arr[4] += n_arr[2]
        data[k_id].append([n_arr[1],n_arr[2],n_arr[3],n_arr[4],n_arr[0],int(arr[0]),int(arr[1])])
        
    print("read data over")
    
    return data

if __name__ =="__main__":
    imgDir = "/home/jjliao/Visdrone_coco/images/val/"
    txtFile = "/data/data/cluster-detector/merge_result.txt"
    
    data = readMergeResult(txtFile)
    
    images = [i for i in os.listdir(imgDir) if '.jpg' in i]
    print('find image', len(images))
    print('find label', len(data.keys()))

    i= 0    
    for k,v in data.items():
        img_id = k
        img = k+".jpg"
        
        imgpath = os.path.join(imgDir, img)
        img_data = cv2.imread(imgpath, -1)  
        print("img path:",imgpath)
        height, width = img_data.shape[:2]  
    
        dot_Visualization(img_data, v,visual_Dir, img)
        
        i+=1
        if i>600:
            break
        