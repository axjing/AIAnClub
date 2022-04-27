
import cv2
import numpy as np
from matplotlib import pyplot as plt

img0=np.ones((512,512),np.uint8)
img0[250:256,250:256],img0[100:106,250:256],img0[250:256,100:106]=0,0,0
dist_transform = cv2.distanceTransform(img0, cv2.DIST_MASK_3, 5)
plt.figure()
plt.subplot(121)
plt.imshow(img0,cmap="gray")
plt.subplot(122)
plt.imshow(dist_transform,cmap="gray")
plt.show()

img = cv2.imread(r'C:\Users\axjing\Downloads\4f48255a239a4c588db4f7270fa32a39.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

ret, thresh50 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
ret, thresh100 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
ret, thresh150 = cv2.threshold(gray, 150, 255, cv2.THRESH_OTSU)
ret, thresh200 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
plt.subplot(231)
plt.imshow(gray,cmap="gray")
plt.subplot(232)
plt.imshow(thresh,cmap="gray")
plt.subplot(233)
plt.imshow(thresh50,cmap="gray")
plt.subplot(234)
plt.imshow(thresh100,cmap="gray")
plt.subplot(235)
plt.imshow(thresh150,cmap="gray")
plt.subplot(236)
plt.imshow(thresh200,cmap="gray")
plt.show()
print(ret)
print(np.unique(thresh))
print(thresh.shape)

# 噪声去除
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# 确定背景区域
sure_bg = cv2.dilate(opening, kernel, iterations=3)  # 膨胀
# 寻找前景区域-对象分离
# separate分离系数，取值范围0.1-1
separate = 0.1
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, separate * dist_transform.max(), 255, 0)  # sure_fg为分离对象的图像
# 找到未知区域
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 类别标记
ret, markers = cv2.connectedComponents(sure_fg)
# 为所有的标记加1，保证背景是0而不是1
markers = markers+1
# 现在让所有的未知区域为0
markers[unknown==255] = 0

markers = cv2.watershed(img, markers)
img[markers == -1] = [0, 255, 0]

dist_transform = cv2.normalize(dist_transform, 0, 1.0, cv2.NORM_MINMAX) * 80

# cv2.imshow("opening", opening)
# cv2.imshow("sure_bg", sure_bg)

# cv2.imshow("dist_transform", dist_transform)

# cv2.imshow("img", img)

# cv2.waitKey()

plt.figure()
plt.subplot(2,3,1)
plt.imshow(thresh,cmap="gray")
plt.subplot(2,3,2)
plt.imshow(opening,cmap="gray")
plt.subplot(2,3,3)
plt.imshow(sure_bg,cmap="gray")
plt.subplot(2,3,4)
plt.imshow(sure_fg,cmap="gray")
plt.subplot(2,3,5)
plt.imshow(dist_transform,cmap="gray")
plt.subplot(2,3,6)
plt.imshow(img)
plt.show()

plt.imshow(img)
plt.show()