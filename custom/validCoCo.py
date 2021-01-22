#coding:utf-8

import json

json_file = 'instances_train.json'
val=json.load(open(json_file, 'r'))
annos = val["annotations"]
imgs = val["images"]

img_dict = {}
for img in imgs:
    k = "{}-{}".format(img["height"],img["width"])
    if k in img_dict.keys():
        img_dict[k] += 1
    else:
        img_dict[k] = 1
    
# for an in annos:
#     if not an["image_id"]==100544:
#         continue
#     print(an["bbox"])

for k in img_dict.keys():
    print("{} - {}".format(k,img_dict[k]))
    
