import numpy as np, sys
import cv2 as cv
import matplotlib.pyplot as plt
#############################################################################
#Watershed 分水岭，重大变化的时间或周期
img = cv.imread('/home/andy/coin.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

#noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel,iterations=2)
#sure background
sure_bg = cv.dilate(opening,kernel,iterations=3)
# distanceTransform()用于计算图像中每一个非零点像素与其最近的零点像素之间的距离，输出
# 的是保存每一个非零点与最近零点的距离信息；可以根据距离变换的这个性质，经过简单的运算，
# 用于细化字符的轮廓和查找物体质心
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret,sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)

ret, markers = cv.connectedComponents(sure_fg)
markers = markers +1
markers[unknown==255] = 0
markers = cv.watershed(img,markers)
img[markers == -1]=[255,255,0]

plt.subplot(221), plt.imshow(markers),plt.xticks([]),plt.yticks([])
plt.subplot(222), plt.imshow(sure_bg),plt.xticks([]),plt.yticks([])
plt.subplot(223), plt.imshow(sure_fg),plt.xticks([]),plt.yticks([])
plt.subplot(224), plt.imshow(img),plt.xticks([]),plt.yticks([])
plt.show()
