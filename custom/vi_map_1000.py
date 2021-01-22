#coding:utf-8

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from Constants import final_detect_result_1000
import json

def print_detection_eval_metrics(coco_eval):
    coco_cls_names = [
        'pedestrian','people','bicycle','car','van',
        'truck','tricycle','awning-tricycle','bus','motor'
    ]
    
    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95

    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                     (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = \
      coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
           '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
    # print("")
    print('MAP:{:.1f}'.format(100 * ap_default))
    for cls_ind, cls in enumerate(coco_cls_names):
        if cls == '__background__':
            continue
        precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind, 0, 2]
        ap = np.mean(precision[precision > -1])
        print(cls+':{:.1f}'.format(100 * ap))

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

resFile = final_detect_result_1000
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

print("------------eval result--------------")
print_detection_eval_metrics(cocoEval)

print("mAP is:{}".format(cocoEval.stats[0]))
print(cocoEval.stats[12:])


