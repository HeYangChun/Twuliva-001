import numpy as np, sys
import cv2 as cv
import matplotlib.pyplot as plt
#############################################################################
#GrabCut algorithm
img = cv.imread('/home/andy/me.jpeg')
plt.imshow(img),plt.colorbar(),plt.show()
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (180,0,1036,662)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()