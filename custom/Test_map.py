#coding:utf-8

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import pylab
from Constants import final_detect_result
import json

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print ('Running demo for *%s* results.'%(annType))

#initialize COCO ground truth api
# dataDir='/home/jjliao/Visdrone_coco/annotations/'
# dataType='val'
# annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)

annFile = "/home/jjliao/Visdrone_coco/annotations/instances_val.json"
cocoGt=COCO(annFile)

#initialize COCO detections api
# resFile='%s/results/%s_%s_fake%s100_results.json'
# resFile = resFile%(dataDir, prefix, dataType, annType)

resFile = final_detect_result
cocoDt=cocoGt.loadRes(resFile)

# imgIds=sorted(cocoGt.getImgIds())
# imgIds=imgIds[0:100]
# imgId = imgIds[np.random.randint(100)]

dts = json.load(open(resFile,'r'))
imgIds = [imid['image_id'] for imid in dts]
imgIds = sorted(list(set(imgIds)))

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
# #只计算某个类别
# cocoEval.params.catIds = [1]
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()


