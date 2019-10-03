import numpy as np, sys
import cv2 as cv
import matplotlib.pyplot as plt
#############################################################################
#Image pyramids
img = cv.imread('/home/andy/me.jpeg',0)
lower_reso = cv.pyrDown(img)
upper_reso = cv.pyrUp(lower_reso)
# Image Blending using Pyramids
A = cv.imread('/home/andy/apple1.png')
B = cv.imread('/home/andy/orange1.png')
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpA.append(G)

G = B.copy()
gpB = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpB.append(G)

lpA=[gpA[5]]
for i in range(5,0,-1):
    GE = cv.pyrUp(gpA[i])
    L  = cv.subtract(gpA[i-1],GE)
    lpA.append(L)

lpB=[gpB[5]]
for i in range(5,0,-1):
    GE = cv.pyrUp(gpB[i])
    L  = cv.subtract(gpB[i-1],GE)
    lpB.append(L)

LS =[]
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    half = int( cols/2)
    ls = np.hstack((la[:,0:half],lb[:,half:]))
    LS.append(ls)

ls_ = LS[0]
for i in range(1,6):
    ls_ =cv.pyrUp(ls_)
    ls_ = cv.add(ls_,LS[i])

real = np.hstack((A[:,:half],B[:,half:]))

cv.imwrite('/home/andy/pr_blending2.jpg',ls_)
cv.imwrite('/home/andy/direct_blending2.jpg',real)