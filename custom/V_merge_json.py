#coding:utf-8
import json
import numpy as np
from Constants import baseDir, merge_result_txt,clip_txt_path

'''
return key-value, filename:box
'''
def readClipTxt(txt_path):
    clip_dict = {}
    f = open(txt_path,"r")
    for line in f.readlines():
        arr = np.array(line.strip('\n').split(","),dtype=np.int32)
        assert not arr[0] in clip_dict.keys()
        clip_dict[arr[0]] = arr[1:5]

    return clip_dict

 
def merge_clip(clip_dict):
    fp = open(baseDir+'sample_results.json', 'r') 
    data = json.load(fp)
    out_f = open(merge_result_txt,"w")

    for bbox in data:
        img_id = int(bbox["image_id"])
        x1,y1,w,h = bbox["bbox"]
        cat_id = bbox["category_id"]
        score = bbox["score"]
        
        if not img_id in clip_dict.keys():
            print("invalid img id:", img_id)
            continue
        
        #将物体坐标映射回大图，即加上每个裁剪区域的左上角xy坐标
        x1 +=  clip_dict[img_id][0]
        y1 +=  clip_dict[img_id][1]
        
        ori_id = str(bbox["image_id"])[:-2]
        
        out_f.write("{},{},{},{},{},{},{}\n".format(ori_id,cat_id,score,x1,y1,w,h))
        
    
    
if __name__=="__main__":
    clip_dict = readClipTxt(clip_txt_path)
    merge_clip(clip_dict)
    