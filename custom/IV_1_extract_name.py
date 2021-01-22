#coding:utf-8
# P02 批量读取文件名（不带后缀）

import os
from Constants import clip_xml_dir,train_name_txt


path_list = os.listdir(clip_xml_dir)  # os.listdir(file)会历遍文件夹内的文件并返回一个列表
fp = open(train_name_txt,"w")

num =0 
for file_name in path_list:
    fp.write(file_name.split(".")[0] + "\n")
    num +=1

fp.close()
print("save to ",train_name_txt)
print("total file:",num)
