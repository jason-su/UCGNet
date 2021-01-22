# -*- coding: UTF-8 -*-
# 聚类代码
import cv2
import os
import sys
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
import matplotlib.pyplot as plt
import time
from Constants import baseDir,box_Dir,erzhimap_Dir,raw_val_img_dir,visual_Dir

'''
第二步：
对二值图进行聚类，输出裁剪的每个区域坐标
例如：输入0001.jpg输出000100.jpg + 000101.jpg
输入：txt文件，每行表示一个坐标点
算法：dbscan和kmeans
输出：txt文件
'''
def dot_Visualization(img_data, lbl_data, save_path, idx):  # 可视化像素点
    w, h = img_data.shape[0], img_data.shape[1]
    image = np.zeros((w, h), np.uint8)
    for xy in lbl_data:
        cv2.circle(img_data, (xy[0],xy[1]), 2, (0,0,255), -1)
    cv2.imwrite(save_path + idx, img_data)
    return img_data


def cluster_method(data, image, idx,out_cluster_path,w,h):  # 聚类操作
    data_xy = []
    c_xys = dbscan(data)
#     c_xys = kmeans(data)
    ih,iw = image.shape[:2]
    for cxy in c_xys:
        cxy = adjustBox(cxy,iw,ih)
        c_x,c_y=cxy[0:2]
#         x1,x2 = max(0,c_x-300),min(c_x+300,w)
#         y1,y2 = max(0,c_y-300),min(c_y+300,h)
#         x1,x2 = max(0,cxy[2]-10),min(cxy[4]+10,w)
#         y1,y2 = max(0,cxy[3]-10),min(cxy[5]+10,h)
        
        x1,y1,x2,y2 = cxy[2],cxy[3],cxy[4],cxy[5]
        
        if x2-x1<50 and y2-y1<50:
            continue
        
        cv2.circle(image, (int(c_x), int(c_y)), 5, (255,255,0), -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 6)
        
        data_xy.append([x1,y1,x2,y2])
    cv2.imwrite(out_cluster_path + idx, image)

    return np.array(data_xy)

def adjustBox(box,iw,ih):
    minw = 600
    minh = 600
    
    w = box[4] - box[2]
    h = box[5] - box[3]
    
    #小于50*50的删掉
    if w<50 and h<50:
        return box
    
#     print("ad pre:",box[0:6],";w:",w,";h:",h)
    
    #调成w，h比例，不要大于1/2
    if w/h >2:
        h = w//2
    if h/w >2:
        w= h//2
    
    if w <minw:
        w = minw
    
    if h<minh:
        h=minh
    
    if box[0]<minw//2:
        box[0] = minw//2
    
    if box[0] > iw-minw//2:
        box[0] = iw-minw//2
    
    if box[1] < minh//2:
        box[1] = minh//2
    if box[1] >ih - minh//2:
        box[1] = ih - minh//2
    
    #更新最后的坐标
    box[2] = max(box[0] - w//2,0)
    box[4] = min(box[0] + w//2,iw)  
    
    box[3] = max(box[1] - h//2,0)
    box[5] = min(box[1] + h//2,ih)   
    
    
#     print("ad af:",box[0:6],";w:",w,";h:",h)
    
    return box   

def kmeans(data):
    n_clu = 6
    labels = KMeans(n_clusters=n_clu, random_state=9).fit_predict(data)
    
    c_xys = []
    
    for k in range(n_clu):
        class_member_mask = (labels == k)
        xy = data[class_member_mask]
        if len(xy) ==0:
            continue
        cx,cy = int(np.mean(xy[:,0])),int(np.mean(xy[:,1]))
        x1,x2 = np.min(xy[:,0]),np.max(xy[:,0])
        y1,y2 = np.min(xy[:,1]),np.max(xy[:,1])
        
        c_xys.append([cx,cy,x1,y1,x2,y2])
    
    return c_xys

def dbscan(data):
    db = DBSCAN(eps=100, min_samples=25).fit(data)  
    
#     db = KMeans(n_clusters=2, random_state=9).fit(data)  
    
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    
#     print("cluster number:{}".format(n_clusters_))
    
    c_xys = []
    
    for k in unique_labels:
        class_member_mask = (labels == k)
        xy = data[class_member_mask & core_samples_mask]
        if len(xy) ==0:
            continue
        cx,cy = int(np.mean(xy[:,0])),int(np.mean(xy[:,1]))
        x1,x2 = np.min(xy[:,0]),np.max(xy[:,0])
        y1,y2 = np.min(xy[:,1]),np.max(xy[:,1])
        
        c_xys.append([cx,cy,x1,y1,x2,y2])
    
    return c_xys


def text_save(filename, data):#filename为写入txt文件的路径，data为要写入数据列表.
    file = open(filename,'w')
    from itertools import chain
    s = '\n'.join([' '.join(chain([filename[-10:-4] + '0' + str(i)], map(str, j))) for i, j in enumerate(data)]) + '\n'
    file.write(s)
    file.close()
    print(" %s 保存文件成功"  %  filename[-10:] ) 


def save_to_txt(imgID,boxes,outDir,w,h,clip_txt_fp):
    file_path = os.path.join(outDir,imgID+"_box.txt")
    file = open(file_path,'w')
    
    out_img_id = ("%s00" % (imgID))
    file.write("{},{},{},{},{}\n".format(out_img_id,0,0,w,h))
    clip_txt_fp.write("{},{},{},{},{}\n".format(out_img_id,0,0,w,h))
        
    for ind,b in enumerate(boxes):
        out_img_id = ("%s%.2d" % (imgID,ind+1))
        file.write("{},{},{},{},{}\n".format(out_img_id,b[0],b[1],b[2],b[3]))
        clip_txt_fp.write("{},{},{},{},{}\n".format(out_img_id,b[0],b[1],b[2],b[3]))
        
    file.close()

def checkPath(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

if __name__ == '__main__':
    # txt存放的路径
#     txt_path = os.path.join(baseDir,"gt_map_copy")
    # 原图片路径 
#     img_path = os.path.join(baseDir,"gt_img_copy")
    # 画出来的图片保存的路径
    checkPath(box_Dir)
    checkPath(visual_Dir)
    out_cluster_path = os.path.join(baseDir,"II_clus_output_copy/")
    checkPath(out_cluster_path)
    
    images = [i for i in os.listdir(raw_val_img_dir) if '.jpg' in i]
    labels = [i for i in os.listdir(erzhimap_Dir) if '.txt' in i]
    print('find image', len(images))
    print('find label', len(labels))

    clip_txt = os.path.join(baseDir,"img_clip.txt")
    clip_txt_fp = open(clip_txt,"w")
    
    
    clus_pic_num = 0
    
#     for idx,lbl in enumerate(labels):
    for idx,img in enumerate(images):
        lbl = img.replace('jpg', 'txt')
        img_id = lbl[:-4]
#         img = lbl.replace('txt', 'jpg')
#         img_data = misc.imread(os.path.join(raw_val_img_dir, img))
        img_data = cv2.imread(os.path.join(raw_val_img_dir, img), -1) 
        
        w,h = img_data.shape[:2]
        
        box = []
        lbl_path = os.path.join(erzhimap_Dir, lbl)
        if os.path.exists(lbl_path):
            lbl_data = np.loadtxt(lbl_path,dtype=np.int32,delimiter=",")
            image = dot_Visualization(img_data, lbl_data, visual_Dir, img)
            if len(lbl_data)==0:
                print("ERROR: empty data:",idx)
            else:
                box = cluster_method(lbl_data, image, img,out_cluster_path,w,h)
        save_to_txt(img_id, box, box_Dir,w,h,clip_txt_fp)
        
        clus_pic_num = clus_pic_num + len(box) + 1
        
#         if clus_pic_num>15:
#             break
        
    clip_txt_fp.close()
    print("All Done, Pic Num:",clus_pic_num)
