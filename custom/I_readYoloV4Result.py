#coding:utf-8

import json
import os
import numpy as np


def readNameId():
    fp = open("name_id_val.txt","r")
    lines = [i.strip('\n').split(",") for i in fp.readlines()]
    data = {}
    for line in lines:
        name,newid = line
        data[name] = newid
    
    return data 

def readResultJson(nameIdArr):
#     json_file = '/data/data/cluster-detector/results_origin.json'
    json_file = '/data/data/cluster-detector/x-big_results_origin.json'
    
    rs=json.load(open(json_file, 'r'))
    
    cat_ids={}
    
    num = len(rs)
    data = {}
    for i in range(num):
        obj = rs[i]
        img_id = obj["image_id"]
        if not img_id in nameIdArr.keys():
            print("can not find:",img_id)
            continue
        new_id = nameIdArr[img_id]
        box = obj["bbox"]
        box = np.array(box).astype(np.int32)
        score = obj["score"]
        cat_ids[obj["category_id"]] = 1
        
        if score <0.3:
            continue
        if not new_id in data.keys():
            data[new_id] = [box]
        else:
            data[new_id].append(box)
    
#    print("cat ids:",cat_ids.keys())
    
    for k in data.keys():
        saveToTxt(k,data[k])
        print("execute "+str(k) +" over")
    
def saveToTxt(img_id,boxes):
    baseDir = "/data/data/cluster-detector/erzhimap-yolov5-big/"
#     baseDir = "c:/tmp/split/"
    if not os.path.exists(baseDir):
        os.makedirs(baseDir)
    boxes = np.array(boxes)
    max_x = np.max(boxes[:,0])
    max_y = np.max(boxes[:,1])
    
    out_txt = os.path.join(baseDir,str(img_id)+".txt")
    
    erzhi_map = np.zeros((max_x+1, max_y+1))
    box_xy = []
    for i, box in enumerate(boxes):
        xtl, ytl = box[0], box[1]
        xbr, ybr = xtl+box[2], ytl + box[3] 
        
        box_xy.append([xtl,ytl,xbr,ybr])

        xtl = int(xtl)
        ytl = int(ytl)
        xbr = int(xbr)
        ybr = int(ybr)
        
        erzhi_map[xtl:xbr,ytl:ybr] = 1 
        
    data_xy = []
    for i_w in range(max_x):
        for j_h in range(max_y):
            if erzhi_map[i_w][j_h] == 1:
                data_xy.append([i_w,j_h])
    
    if len(data_xy) ==0:
        return
    
    data_xy = np.array(data_xy)
    if len(data_xy)>1000:
        sel_mask = np.random.choice(data_xy.shape[0],1000)
        data_xy = data_xy[sel_mask]
                           
    out_fp = open(out_txt,"w")  
    for xy in data_xy:
        out_fp.write("{},{}\n".format(xy[0],xy[1]))  
    
nameIdArr = readNameId()
readResultJson(nameIdArr)

    
