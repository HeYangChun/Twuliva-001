import numpy as np, sys
import cv2 as cv
import matplotlib.pyplot as plt
#############################################################################
# Hough Line Transform,Hough Transform is a popular technique to detect any
# shape, if you can represent that shape in mathematical form. It can detect
# the shape even if it is broken or distorted a little bit
# . Any line can be represented in these two terms, (ρ,θ).
img = cv.imread('/home/andy/sudoku.png')
img2 = img.copy()

gray = cv .cvtColor(img, cv.COLOR_BGR2GRAY)
# Canny 边缘检测算法
edges = cv.Canny(gray, 10, 150, apertureSize = 3, L2gradient=False)

lines = cv.HoughLines(edges, 1, np.pi/360,200)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv.line(img,(x1,y1),(x2,y2),(0,255,255),1)

# edges = cv.Canny(gray,150,150,apertureSize = 3)
# lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
# for line in lines:
#     print(line[0])
#     x1,y1,x2,y2 = line[0]
#     cv.line(img,(x1,y1),(x2,y2),(0,255,0),1)
plt.subplot(121), plt.imshow(edges), plt.xticks([]),plt.yticks([])
plt.subplot(122), plt.imshow(img),  plt.xticks([]),plt.yticks([])
plt.show()
