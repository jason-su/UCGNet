#coding:utf-8
import os
from xml.dom.minidom import Document
import copy
import numpy as np
import cv2
from Constants import raw_val_img_dir,raw_val_ann_dir,box_Dir,clip_img_dir,clip_xml_dir
from matplotlib.pyplot import draw
from scipy import misc
import shutil


'''
第三步：
裁剪图片
输入:
原始图片:raw_val_img_dir
txt标注文件:raw_val_ann_dir
裁剪框文件:box_Dir
输出：每个子图
clip_img_dir,clip_xml_dir
'''
class_list = [
    'ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others'
]


def format_label(txt_list):
    format_data = []
    for i in txt_list[0:]:
        format_data.append([int(xy) for xy in i.split(',')[:8]])
    return np.array(format_data)


def save_to_xml(save_path,
                im_width,
                im_height,
                objects_axis,
                label_name,
                name,
                hbb=True):
    im_depth = 0
    object_num = len(objects_axis)
    doc = Document()

    annotation = doc.createElement('annotataion')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    folder_name = doc.createTextNode('Visdrone')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename_name = doc.createTextNode(name)
    filename.appendChild(filename_name)
    annotation.appendChild(filename)

    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('The Visdrone Database'))
    source.appendChild(database)

    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('Visdrone'))
    source.appendChild(annotation_s)

    image = doc.createElement('image')
    image.appendChild(doc.createTextNode('flickr'))
    source.appendChild(image)

    flickrid = doc.createElement('flickrid')
    flickrid.appendChild(doc.createTextNode('322409915'))
    source.appendChild(flickrid)

    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    flickrid_o = doc.createElement('flickrid')
    flickrid_o.appendChild(doc.createTextNode('knautia'))
    owner.appendChild(flickrid_o)

    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('yang'))
    owner.appendChild(name_o)

    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(im_width)))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(im_height)))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(im_depth)))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)
    for i in range(object_num):
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(
            doc.createTextNode(label_name[int(objects_axis[i][5])]))
        objects.appendChild(object_name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)
        if hbb:
            x0 = doc.createElement('xmin')
            x0.appendChild(doc.createTextNode(str((objects_axis[i][0]))))
            bndbox.appendChild(x0)
            y0 = doc.createElement('ymin')
            y0.appendChild(doc.createTextNode(str((objects_axis[i][1]))))
            bndbox.appendChild(y0)
            x1 = doc.createElement('xmax')
            x1.appendChild(doc.createTextNode(str((objects_axis[i][2]))))
            bndbox.appendChild(x1)
            y1 = doc.createElement('ymax')
            y1.appendChild(doc.createTextNode(str((objects_axis[i][3]))))
            bndbox.appendChild(y1)
        else:

            x0 = doc.createElement('x0')
            x0.appendChild(doc.createTextNode(str((objects_axis[i][0]))))
            bndbox.appendChild(x0)
            y0 = doc.createElement('y0')
            y0.appendChild(doc.createTextNode(str((objects_axis[i][1]))))
            bndbox.appendChild(y0)

            x1 = doc.createElement('x1')
            x1.appendChild(doc.createTextNode(str((objects_axis[i][2]))))
            bndbox.appendChild(x1)
            y1 = doc.createElement('y1')
            y1.appendChild(doc.createTextNode(str((objects_axis[i][3]))))
            bndbox.appendChild(y1)

            x2 = doc.createElement('x2')
            x2.appendChild(doc.createTextNode(str((objects_axis[i][4]))))
            bndbox.appendChild(x2)
            y2 = doc.createElement('y2')
            y2.appendChild(doc.createTextNode(str((objects_axis[i][5]))))
            bndbox.appendChild(y2)

            x3 = doc.createElement('x3')
            x3.appendChild(doc.createTextNode(str((objects_axis[i][6]))))
            bndbox.appendChild(x3)
            y3 = doc.createElement('y3')
            y3.appendChild(doc.createTextNode(str((objects_axis[i][7]))))
            bndbox.appendChild(y3)

    f = open(save_path, 'w')
    f.write(doc.toprettyxml(indent=''))
    f.close()


def clip_image(path_new_xml, img_old, data_new, boxes_all):
    name_new, x1, y1, x2, y2 = data_new
    width, height = x2 - x1, y2 - y1
    
    
    if width<10 or height<10:
        print("ERROR(w,h):{}-{},name:{}".format(width,height,name_new))
        return
    
    if len(boxes_all) > 0:
        # print(width, height)
#         assert (width == 600 and height == 600)
        boxes = copy.deepcopy(boxes_all)
        boxes_new = np.zeros_like(boxes_all)

#         if name_new=="20046704":
#             print("test")
#             import pdb
#             pdb.set_trace()
#         img_new = img_old[x1:x2,y1:y2]
        
        img_new = img_old[y1:y2,x1:x2]
        
        boxes_new[:, 0] = boxes[:, 0] - x1
        boxes_new[:, 0][boxes_new[:, 0]<=0] = 1
        
        boxes_new[:, 2] = boxes[:, 0] + boxes[:, 2] - x1
        boxes_new[:, 2][boxes_new[:, 2]>=width] = width-1
        
        boxes_new[:, 4] = boxes[:, 4]

        boxes_new[:, 1] = boxes[:, 1] - y1
        boxes_new[:, 1][boxes_new[:, 1]<=0] = 1
        
        boxes_new[:, 3] = boxes[:, 1] + boxes[:, 3] - y1
        boxes_new[:, 3][boxes_new[:, 3]>=height] = height-1
        
        boxes_new[:, 5] = boxes[:, 5]

        b_h = (boxes_new[:, 3]-boxes_new[:, 1])
        b_w = (boxes_new[:, 2]-boxes_new[:, 0])

        idx = np.intersect1d(
            np.where(b_h[:] >= 1)[0],
            np.where(b_w[:] >= 1)[0])
#         cond2 = np.intersect1d(
#             np.where(b_h[:] <= (height))[0],
#             np.where(b_w[:] <= (width))[0])
        
#         if name_new == "20001501":
#             import pdb
#             pdb.set_trace()
#             print("hello")
        
#         idx = np.intersect1d(cond1, cond2)
        if len(idx) > 0:
            save_to_xml(path_new_xml, img_new.shape[1], img_new.shape[0], boxes_new[idx,:], class_list,
                                name_new + '.jpg')
            if img_new.shape[0] > 5 and img_new.shape[1] > 5:
                img = os.path.join(clip_img_dir, name_new + '.jpg')
                cv2.imwrite(img, img_new)
        else:
            print("empty dts, name:{}".format(name_new))

if __name__ == '__main__':
    if os.path.exists(clip_img_dir):
        shutil.rmtree(clip_img_dir)#删除再建立
        os.makedirs(clip_img_dir)
    if os.path.exists(clip_xml_dir):
        shutil.rmtree(clip_xml_dir)#删除再建立
        os.makedirs(clip_xml_dir)
        
    print("box_dir:",box_Dir)
    f_num=0
    clip_num = 0
    for i in [i for i in os.listdir(box_Dir) if i[-4:] == '.txt']:
#         print(i)
        f_num +=1
        with open(os.path.join(box_Dir, i), 'r', encoding='utf8') as f:
            lines = [i.split(",") for i in f.readlines()]
            drawer = {}
        
            for line in lines:
                name_new, x1, y1, x2, y2 = line # x1, y1, x2, y2
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                name_old = name_new[:6]
                if name_old in drawer.keys():
                    drawer[name_old].append((name_new, x1, y1, x2, y2))
                else:
                    drawer[name_old] = [(name_new, x1, y1, x2, y2)]
            
            if len(drawer.keys())==0:
                print("error:",name_old)
            
            for name_old, datas in drawer.items():
                path_old = os.path.join(raw_val_img_dir, name_old + '.jpg')
                img_data = cv2.imread(path_old, -1) 
        
        
                txt_data = open(os.path.join(raw_val_ann_dir, name_old + '.txt'),
                                'r').readlines()
                boxes = format_label(txt_data)
                for data_new in datas:
                    name_new, x1, y1, x2, y2 = data_new
                    path_new = os.path.join(clip_img_dir, name_new + '.jpg')
                    path_new_xml = os.path.join(clip_xml_dir,name_new + '.xml')
                    clip_image(path_new_xml, img_data, data_new, boxes)
                    
                    clip_num +=1
    print("file num:{}, clip:{}".format(f_num,clip_num))
    
    
    