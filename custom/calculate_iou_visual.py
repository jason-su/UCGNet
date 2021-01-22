# coding:utf-8
# 计算predict和gt的iou

import numpy as np
import os
import cv2
# from Constants import final_detect_result,VIII_visual_dir
import json
from Constants import final_detect_result,name_id_val_txt

def compute_IOU(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    # import pdb; pdb.set_trace()
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)
        
def readNameId():
    fp = open(name_id_val_txt,"r")
    lines = [i.strip('\n').split(",") for i in fp.readlines()]
    data = {}
    for line in lines:
        name,newid = line
        data[name] = newid
    
    return data 

    
def draw_bboxes(image, bboxes, gt_boxes, font_size=0.5, thresh=0.5, colors=None):
    image = image.copy()
    
    coco_cls_names = [
        'pedestrian','people',
        'bicycle','car','van','truck','tricycle',
        'awning-tricycle','bus','motor'
    ]
    
    p_num = 0
    #draw predict box
    for box in bboxes:
        if box[4] <thresh:
            continue
        # cat_name = coco_cls_names[box[5]-1]
        # cat_size  = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]            
        bbox = np.array([box[0],box[1],box[2],box[3]]).astype(int)
        cv2.rectangle(image,(bbox[0], bbox[1]),(bbox[2], bbox[3]),(0,0,255), 2)
        p_num +=1
    
    print("predict num:",p_num)

    if colors is None:
        color = np.random.random((3, )) * 0.6 + 0.4
        color = (color * 255).astype(np.int32).tolist()
    print("gt num:",len(gt_boxes))
    for gt_box in gt_boxes:
        gt_bbox = np.array([gt_box[0],gt_box[1],gt_box[2],gt_box[3]]).astype(int)
#         import pdb; pdb.set_trace()
        isPredicted = False
        for box in bboxes:
            if box[4] <thresh:
                continue
            # cat_name = coco_cls_names[box[5]-1]
            # cat_size  = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]            
            bbox = np.array([box[0],box[1],box[2],box[3]]).astype(int)
                # iou = compute_IOU(gt_bbox, bbox)
                # import pdb; pdb.set_trace()
#             if box[5] == gt_box[4]:
            iou = compute_IOU(gt_bbox, bbox)
            if iou > 0.5:
                isPredicted = True
                break
        
        if not isPredicted:
            cv2.rectangle(image,(gt_box[0], gt_box[1]),(gt_box[2], gt_box[3]),(0,255,0), 2)
                
        # if bbox[1] - cat_size[1] - 2 < 0:
        #     cv2.rectangle(image,
        #         (bbox[0], bbox[1] + 2),
        #         (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
        #         (255,0,0), -1
        #     )
        #     cv2.putText(image, cat_name,
        #         (bbox[0], bbox[1] + cat_size[1] + 2),
        #         cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
        #     )
        # else:
        #     cv2.rectangle(image,
        #         (bbox[0], bbox[1] - cat_size[1] - 2),
        #         (bbox[0] + cat_size[0], bbox[1] - 2),
        #         (255,0,0), -1
        #     )
        #     cv2.putText(image, cat_name,
        #         (bbox[0], bbox[1] - 2),
        #         cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
        #     )
        # cv2.rectangle(image,
        #     (bbox[0], bbox[1]),
        #     (bbox[2], bbox[3]),
        #     (0,0,255), 2
        # )
    return image

def readFinalResult():
    fp = open(final_detect_result) 
    boxes = json.load(fp)
    
    data = {}
    
    for bbox in boxes:
        img_id = bbox["image_id"]
        x1,y1,x2,y2 = bbox["bbox"]
        cat_id = bbox["category_id"]
        score = bbox["score"]
        
        if not img_id in data.keys():
            data[img_id] = []
        
        data[img_id].append([x1,y1,x1+x2,y1+y2,score,cat_id])

    return data
    
def read_Gt():
    gt_file = open("/home/jjliao/Visdrone_coco/annotations/instances_val.json")
    boxes = json.load(gt_file)
    
    data = {}
    
    for bbox in boxes['annotations']:
        # import pdb; pdb.set_trace()
        img_id = bbox["image_id"]
        x1,y1,x2,y2 = bbox["bbox"]
        cat_id = bbox["category_id"]
        
        if not img_id in data.keys():
            data[img_id] = []
        
        data[img_id].append([x1,y1,x1+x2,y1+y2,cat_id])

    return data

if __name__ =="__main__":
    imgDir = "/home/jjliao/Visdrone_coco/VisDrone2019-DET-val/images/"

    nameIdArr = readNameId()
    # import pdb; pdb.set_trace()
    
    images = [i for i in os.listdir(imgDir) if '.jpg' in i]
    print('find image', len(images))
    
    data = readFinalResult()

    gt_data = read_Gt()

    for idx,img_name in enumerate(images):
#         if idx>10:
#             break
        imgpath = os.path.join(imgDir, img_name)
        img_data = cv2.imread(imgpath, -1)  
        print("img path:",imgpath)
        height, width = img_data.shape[:2]  
        # import pdb; pdb.set_trace()
        newimg = draw_bboxes(img_data, data[int(img_name[:-4])], gt_data[int(img_name[:-4])])
        save_path = os.path.join('/home/jjliao/code/yolov5-x-visdrone/output/predict_gt/',img_name)
        cv2.imwrite(save_path , newimg)
        