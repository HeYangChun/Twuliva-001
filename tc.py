import numpy as np
import cv2 as cv
#############################################################################
#SMOOTH, LPF low pass filter  HPF high pass filter, LPF helps in removing noise
#HPF helps in finding edges
import matplotlib.pyplot as plt
img = cv. imread("/home/andy/opencvlog2.png")

#method1
kernel = np.ones((7,7),np.float32)/(7)
dst1 = cv.filter2D(img,-1,kernel)
#method2
dst2 = cv.blur(img,(5,5))
#method3
dst3 =  cv.GaussianBlur(img,(5,5),0)
#method4
dst4 = cv.medianBlur(img,5)
#method5
dst5 = cv.bilateralFilter(img,9,75,75)

imgs = [img,dst1,dst2,dst3,dst4,dst5]
titles =['org','Filter2D','blur','Gaussion','media','bilater']
# plt.imshow(imgs[0])
for x in range(6):
    plt.subplot(2, 3, x+1), plt.imshow(imgs[x],'gray')
    plt.title(titles[x])
    plt.xticks([])
    plt.yticks([])

plt.show()


