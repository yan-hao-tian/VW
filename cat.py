import os
import cv2
import numpy as np

seg_b3_f = os.listdir('seg_b3_multi-scale/frankfurt')
seg_b3_l = os.listdir('seg_b3_multi-scale/lindau')
seg_b3_m = os.listdir('seg_b3_multi-scale/munster')

seg_b5_f = os.listdir('seg_b5_multi-scale/frankfurt')
seg_b5_l = os.listdir('seg_b5_multi-scale/lindau')
seg_b5_m = os.listdir('seg_b5_multi-scale/munster')

image_f = os.listdir('/data/yht/cityscapes/leftImg8bit/val/frankfurt')
image_l = os.listdir('/data/yht/cityscapes/leftImg8bit/val/lindau')
image_m = os.listdir('/data/yht/cityscapes/leftImg8bit/val/munster')

label_f = os.listdir('/data/yht/cityscapes/gtFine/val/frankfurt')
label_l = os.listdir('/data/yht/cityscapes/gtFine/val/lindau')
label_m = os.listdir('/data/yht/cityscapes/gtFine/val/munster')

for item in seg_b3_f:
    seg_b3_pred_f = cv2.imread('seg_b3_multi-scale/frankfurt/'+item)
    seg_b5_pred_f = cv2.imread('seg_b5_multi-scale/frankfurt/'+item)
    image_ori_f = cv2.imread('/data/yht/cityscapes/leftImg8bit/val/frankfurt/'+item)
    label_ori_f = cv2.imread('/data/yht/cityscapes/gtFine/val/frankfurt/'+item.replace('leftImg8bit', 'gtFine_color'))
    inputs = np.hstack((image_ori_f, seg_b3_pred_f, seg_b5_pred_f, label_ori_f))
    cv2.imwrite('city_comp/frankfurt/'+item, inputs)
    # break

for item in seg_b3_l:
    seg_b3_pred_f = cv2.imread('seg_b3_multi-scale/lindau/'+item)
    seg_b5_pred_f = cv2.imread('seg_b5_multi-scale/lindau/'+item)
    image_ori_f = cv2.imread('/data/yht/cityscapes/leftImg8bit/val/lindau/'+item)
    label_ori_f = cv2.imread('/data/yht/cityscapes/gtFine/val/lindau/'+item.replace('leftImg8bit', 'gtFine_color'))
    inputs = np.hstack((image_ori_f, seg_b3_pred_f, seg_b5_pred_f, label_ori_f))
    cv2.imwrite('city_comp/lindau/'+item, inputs)
    # break

for item in seg_b3_m:
    seg_b3_pred_f = cv2.imread('seg_b3_multi-scale/munster/'+item)
    seg_b5_pred_f = cv2.imread('seg_b5_multi-scale/munster/'+item)
    image_ori_f = cv2.imread('/data/yht/cityscapes/leftImg8bit/val/munster/'+item)
    label_ori_f = cv2.imread('/data/yht/cityscapes/gtFine/val/munster/'+item.replace('leftImg8bit', 'gtFine_color'))
    inputs = np.hstack((image_ori_f, seg_b3_pred_f, seg_b5_pred_f, label_ori_f))
    cv2.imwrite('city_comp/munster/'+item, inputs)
    # break