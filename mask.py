#coding:utf-8
import os
import cv2

def add_mask2image_binary(images_path, masks_path, masked_path):
# Add binary masks to images
    for img_item in os.listdir(images_path):
        print(img_item)
        img_path = os.path.join(images_path, img_item)
        img = cv2.imread(img_path)

        mask_path = os.path.join(masks_path, img_item[:-4]+'.png')  # mask是.png格式的，image是.jpg格式的
        mask = cv2.imread(mask_path)

        masked_seg = cv2.addWeighted(img, 1, mask,0.9,0)

        cv2.imwrite(os.path.join(masked_path, img_item), masked_seg)

images_path = 'D:\VOS\data\DAVIS2017\DAVIS_val\JPEGImages\\480p\\camel'
masks_path = 'F:\Desktop\\17\\camel'
masked_path = './camel/'


if not os.path.exists(masked_path):
    os.makedirs(masked_path)
add_mask2image_binary(images_path, masks_path, masked_path)

