import cv2
import numpy as np

input_image = cv2.imread(r'C:\Users\admin\Documents\work\visionProject\cosmo_test\v2-b40621ccca1d9a432adf1442dcc69540_b.jpg')
print(input_image)
cv2.imshow(winname="input_image", mat=input_image)
# cv2.waitKey()

gaussian_image = cv2.GaussianBlur(input_image , (5,5), 10)
cv2.imshow(winname="gaussian_image", mat=gaussian_image)
# cv2.waitKey()

gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
cv2.imshow(winname="gray_image", mat=gray_image)
# cv2.waitKey()

edge_img = cv2.Canny(gray_image,32,128)
cv2.imshow(winname="edge_img", mat=edge_img)
# cv2.waitKey()

poly_pts = np.array([[[0,368],[300,210],[340,210],[640,368]]])
mask = np.zeros_like(gray_image)
cv2.fillPoly(mask, poly_pts, (122,122,122))
img_mask = cv2.bitwise_and(gray_image, mask)
cv2.imshow(winname="img_mask", mat=img_mask)
cv2.waitKey()