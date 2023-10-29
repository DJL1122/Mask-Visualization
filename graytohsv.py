#coding:utf-8
import os
import cv2

def add_mask2image_binary(seged_path):
# Add binary masks to images
    for img_item in os.listdir(seged_path):
        print(img_item)
        img_path = os.path.join(seged_path, img_item)
        img = cv2.imread(img_path)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imwrite(os.path.join(seged_path, img_item[:-4]+'.png'), hsv)


seg_path = 'D:\VOS\data\DAVIS2017\DAVIS_val\JPEGImages\\480p\judo\\'


add_mask2image_binary(seg_path)

