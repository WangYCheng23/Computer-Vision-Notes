import os, sys
import cv2

img_opencv = cv2.imread(os.path.join(os.getcwd(),'gaoyuanyuan.jpeg'))
img_dimensions = img_opencv.shape
total_number_of_elements= img_opencv.size
image_dtype = img_opencv.dtype
print(img_dimensions, total_number_of_elements, image_dtype)
cv2.imshow("original image", img_opencv)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray_img = cv2.imread(os.path.join(os.getcwd(),'gaoyuanyuan.jpeg'), cv2.IMREAD_GRAYSCALE)
dimensions = gray_img.shape
print(dimensions)
cv2.imshow("gray image", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

b,g,r = cv2.split(img_opencv)
img_matplotlib = cv2.merge([r,g,b])

from matplotlib import pyplot as plt
plt.subplot(121)
plt.imshow(img_opencv)
plt.subplot(122)
plt.imshow(img_matplotlib)
plt.show()

# cv2.imshow('bgr image', img_opencv)
# cv2.imshow('rgb image', img_matplotlib)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import numpy as np
img_concats = np.concatenate((img_opencv, img_matplotlib), axis=1)
cv2.imshow('bgr image and rgb image', img_concats)
cv2.waitKey(0)
cv2.destroyAllWindows()
