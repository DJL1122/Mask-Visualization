# 将分割图和原图合在一起
from PIL import Image
import matplotlib.pyplot as plt

# image1 原图
# image2 分割图
image1 = Image.open("D:\VOS\data\DAVIS2017\DAVIS_val\JPEGImages\\480p\camel\\00040.jpg")
image2 = Image.open("D:\VOS\data\DAVIS2017\DAVIS_val\Annotations\\480p\camel\\00040.png")

image1 = image1.convert('RGBA')
image2 = image2.convert('RGBA')

# 两幅图像进行合并时，按公式：blended_img = img1 * (1 – alpha) + img2* alpha 进行
image = Image.blend(image1, image2, 1)
image.save("test.png")
