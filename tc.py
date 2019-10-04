import numpy as np, sys
import cv2 as cv
import matplotlib.pyplot as plt
#############################################################################
#contour
img = cv .imread('/home/andy/apple.png')
imggray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret,thresh = cv.threshold(imggray,144,255,0)
#old ver im2,contours,hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours,hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#draw contour
#draw all contour
cv.drawContours(img,contours,-1,(0,0,255), 1)
#draw an individual contour, ex, 4th
# cv.drawContours(img,contours, 9, (0,0,255), 1)
#useful
# cnt = contours[4]
# cv.drawContours(imggray,[cnt], 0, (0,255,0),3)
cv.imshow('image and contours',img)
cv.waitKey(0)
cv.destroyAllWindows()