#coding:utf-8
import numpy as np
import os
import cv2
from Constants import final_detect_result,VIII_visual_dir
import json

    
def draw_bboxes(image, bboxes, font_size=0.5, thresh=0.5, colors=None):
    image = image.copy()
    
    coco_cls_names = [
        'pedestrian','people',
        'bicycle','car','van','truck','tricycle',
        'awning-tricycle','bus','motor'
    ]

    if colors is None:
        color = np.random.random((3, )) * 0.6 + 0.4
        color = (color * 255).astype(np.int32).tolist()

    for box in bboxes:
        if box[4] <thresh:
            continue
        print(box[4])
        cat_name = coco_cls_names[box[5]-1]
        cat_size  = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]
        
        
        bbox = np.array([box[0],box[1],box[2],box[3]]).astype(int)
        if bbox[1] - cat_size[1] - 2 < 0:
            cv2.rectangle(image,
                (bbox[0], bbox[1] + 2),
                (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                color, -1
            )
            cv2.putText(image, cat_name,
                (bbox[0], bbox[1] + cat_size[1] + 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
            )
        else:
            cv2.rectangle(image,
                (bbox[0], bbox[1] - cat_size[1] - 2),
                (bbox[0] + cat_size[0], bbox[1] - 2),
                color, -1
            )
            cv2.putText(image, cat_name,
                (bbox[0], bbox[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
            )
        cv2.rectangle(image,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color, 2
        )
    return image

def readFinalResult():
    fp = open(final_detect_result) 
    boxes = json.load(fp)
    
    data = {}
    
    for bbox in boxes:
        img_id = int(bbox["image_id"])
        x1,y1,x2,y2 = bbox["bbox"]
        cat_id = bbox["category_id"]
        score = bbox["score"]
        
        if not img_id in data.keys():
            data[img_id] = []
        
        data[img_id].append([x1,y1,x1+x2,y1+y2,score,cat_id])
    
    return data

if __name__ =="__main__":
    imgDir = "/home/jjliao/Visdrone_coco/images/val/"
    
    images = [i for i in os.listdir(imgDir) if '.jpg' in i]
    print('find image', len(images))
    
    data = readFinalResult()

    
    for idx,img_name in enumerate(images):
        if idx>550:
            break
        
        imgpath = os.path.join(imgDir, img_name)
        img_data = cv2.imread(imgpath, -1)  
        print("img path:",imgpath)
        height, width = img_data.shape[:2]  
        
        newimg = draw_bboxes(img_data, data[int(img_name[:-4])])
        save_path = os.path.join(VIII_visual_dir,img_name)
        cv2.imwrite(save_path , newimg)
        