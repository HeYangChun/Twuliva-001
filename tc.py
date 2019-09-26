import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt
#Imgae threashold

img = cv.imread('/home/andy/sudoku.png',0)
# img = cv.medianBlur(img,5)
ret, th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,    cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
titles = ['Org','Glb','ApaM','ApaG']
imgs = [img, th1,th2,th3]
for i in range(4):
    plt.subplot(2,3,i+1),plt.imshow(imgs[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
exit()

#if val > threshold-h val=high else val = valTBD
# diffenec betwwen these effects
img = cv.imread('/home/andy/gradient.jpg')
ret, thresh1 = cv.threshold(img,72,96,cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(img,72,96,cv.THRESH_BINARY_INV)
ret, thresh3 = cv.threshold(img,72,96,cv.THRESH_TRUNC)
ret, thresh4 = cv.threshold(img,72,96,cv.THRESH_TOZERO)
ret, thresh5 = cv.threshold(img,72,96,cv.THRESH_TOZERO_INV)

titles = ['Org','Bin','IBin','Trc','TZero','ITZero']
imgs = [img,thresh1,thresh2,thresh3,thresh4,thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(imgs[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()