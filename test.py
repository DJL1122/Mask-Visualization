from __future__ import division


# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time

import os
import argparse
import copy



import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

images_path = 'D:\VOS\data\DAVIS2017\DAVIS_val\JPEGImages\\480p\judo'
masks_path = 'F:\Desktop\DAVIS\\frtm_17\dv2017val-shuiyuan_ep0257\judo'

palette = [
    0, 0, 0,
    0.5020, 0, 0,
    0, 0.5020, 0,
    0.5020, 0.5020, 0,
    0, 0, 0.5020,
    0.5020, 0, 0.5020,
    0, 0.5020, 0.5020,
    0.5020, 0.5020, 0.5020,
    0.2510, 0, 0,
    0.7529, 0, 0,
    0.2510, 0.5020, 0,
    0.7529, 0.5020, 0,
    0.2510, 0, 0.5020,
    0.7529, 0, 0.5020,
    0.2510, 0.5020, 0.5020,
    0.7529, 0.5020, 0.5020,
    0, 0.2510, 0,
    0.5020, 0.2510, 0,
    0, 0.7529, 0,
    0.5020, 0.7529, 0,
    0, 0.2510, 0.5020,
    0.5020, 0.2510, 0.5020,
    0, 0.7529, 0.5020,
    0.5020, 0.7529, 0.5020,
    0.2510, 0.2510, 0]
palette = (np.array(palette) * 255).astype('uint8')



def overlay_davis(image,mask,colors=[255,0,0],cscale=2,alpha=0.4):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # import skimage
    from scipy.ndimage.morphology import binary_erosion, binary_dilation

    colors = np.reshape(colors, (-1, 3))
    #colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)


    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        print(object_id)
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[1])
        binary_mask = mask == 75

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours,:] = 0

    return im_overlay.astype(image.dtype)

def add_mask2image_binary(images_path, masks_path, masked_path):
    # Add binary masks to images
    for img_item in os.listdir(images_path):
        print(img_item)
        img_path = os.path.join(images_path, img_item)
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)

        mask_path = os.path.join(masks_path, img_item[:-4] + '.png')  # mask是.png格式的，image是.jpg格式的
        # segs_path = os.path.join(seged_path, img_item[:-4]+'.png')

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


        canvas = overlay_davis(img, mask, palette)
        canvas = Image.fromarray(canvas)
        canvas.save(os.path.join(masked_path, 'f{}.jpg'.format(img_item)))




images_path = 'D:\VOS\data\DAVIS2017\DAVIS_val\JPEGImages\\480p\judo'
masks_path = 'F:\Desktop\DAVIS\\frtm_17\dv2017val-shuiyuan_ep0257\judo'
masked_path = './4/'

if not os.path.exists(masked_path):
    os.makedirs(masked_path)

add_mask2image_binary(images_path, masks_path, masked_path)

