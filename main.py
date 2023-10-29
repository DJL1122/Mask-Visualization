# coding:utf-8
import os
import cv2


def add_mask2image_binary(images_path, masks_path, masked_path):
    # Add binary masks to images
    for img_item in os.listdir(images_path):
        print(img_item)
        img_path = os.path.join(images_path, img_item)
        img = cv2.imread(img_path)

        mask_path = os.path.join(masks_path, img_item[:-4] + '.png')  # mask是.png格式的，image是.jpg格式的
        # segs_path = os.path.join(seged_path, img_item[:-4]+'.png')

        mask = cv2.imread(mask_path)
        # seg = cv2.imread(segs_path)

        masked_gt = cv2.addWeighted(img,0.8, mask, 1, 3)
        # masked_seg = cv2.addWeighted(img, 1, seg, 1.2, 0)

        cv2.imwrite(os.path.join(masked_path, img_item), masked_gt)
        # cv2.imwrite(os.path.join(masked_path, img_item[:-4]+'seg'+'.jpg'), masked_seg)

#G:\STM_V\\17\camel  D:\Download\DAVIS17_validation_s03\DAVIS17_validation_s03  F:\Desktop\DAVIS\\frtm_17\dv2017val-shuiyuan_ep0257
images_path = 'D:\VOS\data\DAVIS2017\DAVIS_val\JPEGImages\\480p\\soapbox'
masks_path = 'D:\Download\DAVIS17_validation_s03\DAVIS17_validation_s03\\soapbox'
masked_path = './soapbox'

if not os.path.exists(masked_path):
    os.makedirs(masked_path)
add_mask2image_binary(images_path, masks_path, masked_path)
