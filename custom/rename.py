#coding:utf-8
import os

def rename():
    path = "/home/jjliao/Visdrone_coco/VisDrone2019-DET-train/images_origin"
    new_path = "/home/jjliao/Visdrone_coco/VisDrone2019-DET-train/images"
    filelist = os.listdir(path) #该文件夹下所有的文件（包括文件夹）
    
    
    filelist.sort(key=lambda x:x.split('.')[0])
    count = 200001
    for file in filelist:
        print(file)
    for file in filelist:   #遍历所有文件
        Olddir=os.path.join(path,file)   #原来的文件路径
        if os.path.isdir(Olddir):   #如果是文件夹则跳过
            continue
        filename=os.path.splitext(file)[0]   #文件名
        filetype=os.path.splitext(file)[1]   #文件扩展名
        Newdir=os.path.join(new_path,str(count)+filetype)  #用字符串函数zfill 以0补全所需位数
        os.rename(Olddir,Newdir)#重命名
        count+=1

def genID_NameTxt():
    path = "/home/jjliao/Visdrone_coco/VisDrone2019-DET-val/images_origin"
    filelist = os.listdir(path) #该文件夹下所有的文件（包括文件夹）
    
    out_f = open("/data/data/cluster-detector/name_id_val.txt","w")
    
    filelist.sort(key=lambda x:x.split('.')[0])
    #train data
#     count = 100001
    #val start 
    count = 200001
    for file in filelist:   #遍历所有文件
        print("file:",file)
        Olddir=os.path.join(path,file)   #原来的文件路径
        if os.path.isdir(Olddir):   #如果是文件夹则跳过
            continue
        filename=os.path.splitext(file)[0]   #文件名
#         filetype=os.path.splitext(file)[1]   #文件扩展名
        out_f.write(filename+","+str(count)+"\n")
        count+=1
        
genID_NameTxt()