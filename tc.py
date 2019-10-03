import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
#############################################################################
#Image gradient
#Sobel 像素图像边缘检测　算子　索贝尔
img = cv.imread('/home/andy/sudoku.png',0)
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0, ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1, ksize=5)

imgs = [img,laplacian,sobelx,sobely]
title = ["orig",'laplacian','sobelx','sobely']
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(imgs[i],cmap='gray')
    plt.title(title[i])
    plt.xticks([])
    plt.yticks([])

plt.show()