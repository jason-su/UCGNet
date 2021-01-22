
#coding:utf-8
import os
from xml.dom.minidom import Document
import copy
import numpy as np
from scipy import misc
import cv2
from Constants import raw_val_img_dir, raw_val_ann_dir, temp_txt_dir, baseDir
import pdb

'''
第一步，若图像长度超过1000像素，则裁剪成两张图片；若不超过则使用原图；
第二步，裁剪后的图片加上原图，一共三张图片;
第三步，用这些图片进行III,IV,V,VI,VII的过程. 

下面代码是裁剪出超过1000像素的图片，保存为txt格式，便于后续裁剪出image和xml.
'''

def save_to_txt(imgID,boxes,outDir,w,h,clip_txt_fp):
    file_path = os.path.join(outDir,imgID+"_box.txt")
    file = open(file_path,'w')
    out_img_id = ("%s00" % (img[:-4]))
    file.write("{},{},{},{},{}\n".format(out_img_id,0,0,w,h))
    clip_txt_fp.write("{},{},{},{},{}\n".format(out_img_id,0,0,w,h))
        
    for ind, b in enumerate(boxes):
        out_img_id = ("%s%.2d" % (img[:-4],ind+1))
        file.write("{},{},{},{},{}\n".format(out_img_id,b[0],b[1],b[2],b[3]))
        clip_txt_fp.write("{},{},{},{},{}\n".format(out_img_id,b[0],b[1],b[2],b[3]))
        
    file.close()

def save_to_origin_txt(imgID,outDir,w,h,clip_txt_fp):
    file_path = os.path.join(outDir,imgID+"_box.txt")
    file = open(file_path,'w')
    out_img_id = ("%s00" % (img[:-4]))
    file.write("{},{},{},{},{}\n".format(out_img_id,0,0,w,h))
    clip_txt_fp.write("{},{},{},{},{}\n".format(out_img_id,0,0,w,h))
    file.close()

if __name__ == '__main__':
    print("raw_val_img_dir:",raw_val_img_dir)
    print("raw_val_ann_dir:",raw_val_ann_dir)
    images = [i for i in os.listdir(raw_val_img_dir) if 'jpg' in i]
    print('find image', len(images))
    crop_h, crop_w, stride_h, stride_w = 1000, 1000, 1000, 1000
    clip_txt = os.path.join(baseDir, "img_1000_clip.txt")
    clip_txt_fp = open(clip_txt, "w")
    for idx, img in enumerate(images):
        path_old = os.path.join(raw_val_img_dir, img)
        img_data = cv2.imread(os.path.join(raw_val_img_dir, img), -1) 
        image_h, image_w = img_data.shape[:2]
        if image_w > crop_w or image_h > crop_h:
            box = []
            box1 = [0, 0, int(image_w / 2), image_h]
            box2 = [int(image_w / 2), 0, image_w, image_h]
            box.append(box1)
            box.append(box2)
            save_to_txt(img[:-4], box, temp_txt_dir, image_w, image_h, clip_txt_fp)
        else:
            save_to_origin_txt(img[:-4], temp_txt_dir, image_w, image_h, clip_txt_fp)
    
    clip_txt_fp.close()
    print("All done.")

