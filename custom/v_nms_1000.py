#coding:utf-8
import numpy as np
from Constants import merge_result_txt_1000, final_detect_result_1000
import json


def nms(dets, thresh):
#     print("dets shape:",dets.shape)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def readMergeResult():
    data = {}
    f = open(merge_result_txt_1000,"r")
#     f = open("merge_result.txt","r")
    i=0
    for line in f.readlines():
        arr = np.array(line.strip('\n').split(","))
        k_id = ("{}-{}".format(arr[0],arr[1]))
        if not k_id in data.keys():
            data[k_id] = []
        n_arr = np.array(arr[2:7],dtype=np.float32)
        n_arr[3] += n_arr[1]
        n_arr[4] += n_arr[2]
        data[k_id].append([n_arr[1],n_arr[2],n_arr[3],n_arr[4],n_arr[0],int(arr[0]),int(arr[1])])
        
        i +=1
        
    print("read data over")
    
    return data
    
def exeNMS(data):
    ori_num,af_num =0,0
    for k,v in data.items():
        ori_num += len(v)
        v1 = np.array(v)
        r = nms(v1,0.5)
        af_num += len(r)
        if len(r)==1:
            data[k] = [v1[r[0]]]
        else:
            data[k] = v1[r]
        
    print("exe nms over:bf,{}-af,{}".format(ori_num,af_num))
    return data

def saveToJson(rs_data):
    out_data = []
    for k,vs in rs_data.items():
        for v in vs:
            obj = {}
            obj["bbox"]=[v[0],v[1],v[2]-v[0],v[3]-v[1]]
            obj["category_id"]=int(v[6])
            obj["image_id"] = int(v[5])
            obj["score"] = v[4]
            out_data.append(obj)
    
#     f = open("final_detect_result.json","w")
    f = open(final_detect_result_1000,"w")
    json.dump(out_data,f)    
    print("save file to:",final_detect_result_1000)

data = readMergeResult()
rs_data = exeNMS(data)
saveToJson(rs_data)
        

