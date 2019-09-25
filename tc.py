import numpy as np
import cv2 as cv

def showImage(title, img):
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

img = cv.imread('/home/andy/1.jpeg')
res = cv.resize(img, None, fx=0.1, fy=0.1, interpolation=cv.INTER_CUBIC)
# showImage('res',res)
# or
height, width = img.shape[:2]
res = cv.resize(img, (int(0.2 * width), int(0.2 * height)), interpolation=cv.INTER_CUBIC)
# showImage('res',res)

rows, cols = res.shape[:2]
# M = [1 0 100
#     0 1 50]
M = np.float32([[0.8, 0, 10], [0, 0.8, 10]])
dst = cv.warpAffine(res, M, (cols, rows))
# showImage('dst',dst)

# rotation
# M= [cosA   -sinA
#    sinA   cosA]
M = cv.getRotationMatrix2D((cols / 2, rows / 2), 360, 0.6)
dst = cv.warpAffine(res, M, (cols, rows))
# showImage('dst',dst)

# affine transformation
from matplotlib import pyplot as plt

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv.getAffineTransform(pts1, pts2)
dst = cv.warpAffine(res, M, (cols, rows))
#why color changed using below plt.imshow()?
# plt.subplot(121), plt.imshow(res), plt.title('input')
# plt.subplot(122), plt.imshow(dst), plt.title('output')
# plt.show()
# showImage('input',res)
# showImage('output',dst)

#
rows, cols, ch = res.shape
pts1 = np.float32([[56, 65], [368, 52], [28, 387],[389,390]])
pts2 = np.float32([[ 0,  0], [300,  0], [ 0, 300],[300,300]])
M = cv.getPerspectiveTransform(pts1, pts2)
dst = cv.warpPerspective(res, M, (300, 300))
showImage('input',res)
showImage('output',dst)
