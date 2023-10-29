import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import os

image_root = Path('D:\VOS\data\DAVIS2017\DAVIS_val\JPEGImages\\480p\judo')
mask_root = Path('D:\VOS\data\DAVIS2017\DAVIS_val\Annotations\\480p\judo')
save_root = Path('./4/')

# for mask in tqdm(mask_root.iterdir()):
for img_item in os.listdir(image_root):

    imagepath = os.path.join(image_root, img_item)
    mask = os.path.join(mask_root, img_item[:-4] + '.png')
    # mask = Image.open(mask)
    # image = Image.open(imagepath)
    # print(mask.mode)        # L
    # print(image.mode)       # RGB
    mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(imagepath), cv2.IMREAD_COLOR)
    print(mask.shape)       # 1080 1920
    # print(image.shape)      # 1080 1920 3

    image = image.astype(np.float64)
    image[mask > 100] = (image[mask > 100] * 0.6).astype(np.int64)
    image[mask > 100] += np.array([100, 0, 0], dtype=np.int64)