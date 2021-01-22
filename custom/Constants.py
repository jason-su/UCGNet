#coding:utf-8
import os

def checkPath(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
# -------------------------big       
#中间数据存放的基础路径
baseDir = "/data/data/cluster-detector/"
name_id_val_txt = baseDir+"name_id_val.txt"
#第一步输出：二值网络，并且经过采样后留下的二值点，一张图片对应一个txt文件
erzhimap_Dir = baseDir + "erzhimap-yolov5-big/"
#第二步输出：二值网络，并且经过采样后留下的二值点，一张图片对应一个txt文件
box_Dir = baseDir + "II_clusbox_txt/"
visual_Dir = baseDir + "II_visual_dir/"
clip_txt_path = baseDir +"img_clip.txt"
#第三步输出:裁剪后的图片和xml
clip_img_dir = os.path.join(baseDir, 'images/train/')
clip_xml_dir = os.path.join(baseDir, 'xmls/train/')
train_txt_dir = os.path.join(baseDir, 'labels/train/')
#第六步输出：目标框，json格式
final_detect_result=baseDir + "final_results.json"
#yolo输出的路径
yolo_result_dir = os.path.join(baseDir, 'yolo_out/')

checkPath(visual_Dir)
checkPath(clip_img_dir)
checkPath(clip_xml_dir)
checkPath(train_txt_dir)

#第四步1的输出：1个txt文件
train_name_txt = os.path.join(baseDir, 'IV_1_train_name.txt')
#第四步2的输出：1个train_txt文件
yolo_trainval_txt = os.path.join(baseDir, 'trainval.txt')
yolo_trainval_dir = os.path.join(baseDir, 'clip_yolo/')
#第五步的输出：1个result_txt文件
merge_result_txt = os.path.join(baseDir, 'merge_result.txt')

#第8步：画检测框
VIII_visual_dir = os.path.join(baseDir, 'detect_box_visual/')
checkPath(VIII_visual_dir)

#原始数据基础路径
raw_data_dir = "/home/jjliao/Visdrone_yolo_cluster/"

#原始图片存放目录
raw_val_dir = raw_data_dir + "VisDrone2019-DET-val/"
raw_train_img_dir = raw_data_dir + "VisDrone2019-DET-train/"
raw_val_img_dir = raw_val_dir + "images/"
raw_val_ann_dir = raw_val_dir + "annotations/"
# baseDir = "/home/jjliao/cluster"
# baseDir = "C:/tmp/cluster"
raw_data = "/home/jjliao/Visdrone_yolo_cluster/VisDrone2019-DET-train/"
cluster_out_dir = "/home/jjliao/cluster/cluster_output/"

# -------------------------------crop 1000
# 第一步：原始标签数据中转（超过1000裁剪和原图合并）
temp_txt_dir = raw_val_dir + "temp_annot/"
# 第二步：图片超过1000的裁剪后目录
crop_1000_img_dir = os.path.join(baseDir, 'images_1000/train/') 
crop_1000_xml_dir = os.path.join(baseDir, 'xmls_1000/train/') 
# 第三步1的输出：1个txt文件
train_name_txt_1000 = os.path.join(baseDir, 'iii_1_train_name.txt')
# 第三步2的输出：1个train_txt文件
yolo_trainval_txt_1000 = os.path.join(baseDir, 'iii_trainval.txt')
train_txt_dir_1000 = os.path.join(baseDir, 'labels_1000/train/')
# 第四步：1个result_txt文件
merge_result_txt_1000 = os.path.join(baseDir, 'merge_result_1000.txt')
clip_txt_path_1000 = baseDir + "img_1000_clip.txt"
# 第五步：目标框，json格式
final_detect_result_1000 = baseDir + "final_results_1000.json"
# 第六步：计算map

# 第七步：画检测框
VIII_visual_dir_1000 = os.path.join(baseDir, 'detect_box_visual_1000/')
checkPath(VIII_visual_dir_1000)

# ---------------------------------------600
#中间数据存放的基础路径
baseDir = "/data/data/cluster-detector/"
#第一步输出：二值网络，并且经过采样后留下的二值点，一张图片对应一个txt文件
erzhimap_Dir_600 = baseDir + "erzhimap-yolov5-600/"
#第二步输出：二值网络，并且经过采样后留下的二值点，一张图片对应一个txt文件
box_Dir_600 = baseDir + "II_clusbox_txt_600/"
visual_Dir_600 = baseDir + "II_visual_dir_600/"
clip_txt_path_600 = baseDir +"img_clip_600.txt"
#第三步输出:裁剪后的图片和xml
clip_img_dir_600 = os.path.join(baseDir, 'images_600/train/')
clip_xml_dir_600 = os.path.join(baseDir, 'xmls_600/train/')
train_txt_dir_600 = os.path.join(baseDir, 'labels_600/train/')
#第六步输出：目标框，json格式
final_detect_result_600 = baseDir + "final_results_600.json"
#yolo输出的路径
yolo_result_dir_600 = os.path.join(baseDir, 'yolo_out_600/')

checkPath(visual_Dir_600)
checkPath(clip_img_dir_600)
checkPath(clip_xml_dir_600)
checkPath(train_txt_dir_600)

# ---------------------------------------uavdt-----------------------
#中间数据存放的基础路径
baseDir = "/data/data/cluster-detector/"
#第一步输出：二值网络，并且经过采样后留下的二值点，一张图片对应一个txt文件
erzhimap_Dir_uavdt = baseDir + "erzhimap_uavdt/"
#第二步输出：二值网络，并且经过采样后留下的二值点，一张图片对应一个txt文件
box_Dir_uavdt = baseDir + "2_clusbox_txt_uavdt/"
visual_Dir_uavdt = baseDir + "2_visual_dir_uavdt/"
clip_txt_path_uavdt = baseDir +"img_clip_uavdt.txt"
#第三步输出:裁剪后的图片和xml
clip_img_dir_uavdt = os.path.join(baseDir, 'images_uavdt/train/')
clip_xml_dir_uavdt = os.path.join(baseDir, 'xmls_uavdt/train/')
train_txt_dir_uavdt = os.path.join(baseDir, 'labels_uavdt/train/')

#yolo输出的路径
yolo_result_dir_uavdt = os.path.join(baseDir, 'yolo_out_uavdt/')

checkPath(visual_Dir_uavdt)
checkPath(clip_img_dir_uavdt)
checkPath(clip_xml_dir_uavdt)
checkPath(train_txt_dir_uavdt)

#第四步1的输出：1个txt文件
train_name_txt_uavdt = os.path.join(baseDir, '4_1_train_name_uavdt.txt')
#第四步2的输出：1个train_txt文件
yolo_trainval_txt_uavdt = os.path.join(baseDir, 'trainval_uavdt.txt')
yolo_trainval_dir_uavdt = os.path.join(baseDir, 'clip_yolo_uavdt/')
#第五步的输出：1个result_txt文件
merge_result_txt_uavdt = os.path.join(baseDir, 'merge_result_uavdt.txt')
#第六步输出：目标框，json格式
final_detect_result_uavdt = baseDir + "final_results_uavdt.json"
#第8步：画检测框
VIII_visual_dir_uavdt = os.path.join(baseDir, 'detect_box_visual_uavdt/')
checkPath(VIII_visual_dir_uavdt)

#uavdt原始数据基础路径
raw_data_dir_uavdt = "/data/data/UAVDT_coco/"

#原始图片存放目录
raw_val_dir_uavdt = raw_data_dir_uavdt + "images/"
raw_train_img_dir_uavdt = raw_val_dir_uavdt + "train/"
raw_val_img_dir_uavdt = raw_val_dir_uavdt + "val/"
raw_val_ann_dir_uavdt = "/data/data/UAVDT/annotations/"

