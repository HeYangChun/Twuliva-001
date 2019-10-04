import numpy as np, sys
import cv2 as cv
import matplotlib.pyplot as plt
#############################################################################
#Histogram
# an overall idea about the intensity distribution of an imag
img = cv .imread('/home/andy/apple1.png',0)
hist, bin = np.histogramk(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img2 = cdf[img]
plt.subplot(131),plt.imshow(img)
plt.subplot(132),plt.imshow(img2)
plt.subplot(133)
plt.plot(cdf_m, color = 'b')
plt.hist(img2.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

equ = cv.equalizeHist(img)
res = np.hstack((img,equ))
cv.imwrite('/home/andy/res.png',res)
#CLAHE
img = cv.imread('/home/andy/tsukuba_l.png',0)
clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
cl1 = clahe.apply(img)
plt.subplot(121), plt.imshow(img,cmap='gray')
plt.subplot(122), plt.imshow(cl1,cmap='gray')
plt.show()
cv.imwrite('/home/andy/tsu2.png',cl1)