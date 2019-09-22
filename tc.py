import numpy as np
import cv2 as cv

img = cv.imread('/home/andy/1.jpeg')
#px is like this:[b,g,r]
px = img[100,100]
#access only blue
blue = img[100,100,0]
#modify pixel
img[100,100] = [255,255,255]
#array method item and itemset
img.item(10,10,2)
img.itemset((10,10,2),100)
#image attribute
img.shape
img.size
img.dtype
#ROI Region of Intresting
# anything = img[300:400,200:400]
# img[0:100, 0:200] = anything
#split color channel
b,g,r = cv.split(img)
img = cv.merge((r,g,b))
#or
b = img[:,:,0]

cv.imshow("WINDOW TITLE",img)
cv.waitKey(0)
cv.destroyAllWindows()