import numpy as np
import cv2 as cv
#############################################################################
#load image
img = cv.imread("PIC URL",0)
#show
cv.imshow("WINDOW TITLE",img)
cv.waitKey(0)
cv.destroyAllWindows()

#############################################################################
#save image
cv.imwrite("FILENAME",img)
#using matplotlib
from matplotlib import pyplot as plt
plt.imshow(img,cmap='gray',interploation='bicubic')
plt.xticks([]), plt.yticks([]) #hide tick values on X and Y axis
plt.show()

#############################################################################
#capture a video
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
cap.set(cv.CAP_PROP_FRAME_WIDTH,1024)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,768)
while True:
    #capture frame-by-frame
    ret,frame = cap.read()
    if not ret:
        print("Cannot receive frame,exiting")
        break
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    frame = cv.rotate(frame,cv.ROTATE_180)
    cv.imshow("frame",frame)
    if cv.waitKey(1) == ord('q'):
        break;
cap.release()
cv.destroyAllWindows()

#############################################################################
#saving a video
cap = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi',fourcc,20.0,(640,480))
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        print("Cannot receive frame, exiting")
        break
    # flipcode: >0: y axis; =0: x axis; <0: both
    frame=cv.flip(frame,1)
    # frame = cv.rotate(frame,cv.ROTATE_90_COUNTERCLOCKWISE)
    out.write(frame)
    cv.imshow('frame',frame)
    if cv.waitKey(1) == ord('q'):
        break;
cap.release()
out.release()
cv.destroyAllWindows()

#############################################################################
#Draw something
img = np.zeros((512,512,3),np.uint8)
#draw a blue BGR(255,0,0) line with thickness of 5 px
cv.line(img,(0,0),(511,511),(255,0,0),5)
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
cv.circle(img,(63,63),63,(0,0,255),-1)
cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img,[pts],True,(0,255,255))
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'HeYC',(10,500),font,4,(255,255,255),2,cv.LINE_4)
cv.imshow("WINDOW TITLE",img)
cv.waitKey(0)
cv.destroyAllWindows()

#############################################################################
#Event handle, callback functions
#List events information
# events = [i for i in dir(cv) if 'EVENT' in i]
drawing = False
mode = True
ix, iy = -1, -1

#a Callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv.circle(img, (x, y), 20, (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        else:
            cv.circle(img, (x, y), 20, (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img, (x, y), 100, (0, 0, 255), -1)


img = np.zeros((512, 512, 3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image', draw_circle)

while True:
    cv.imshow('image', img)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break;
    if cv.waitKey(20) & 0xFF == ord('m'):
        mode = not mode
cv.destroyAllWindows()

#############################################################################
#Trackbar in CV
def nothing(x):
    pass

img = np.zeros((300,512,3),np.uint8)
cv.namedWindow('image')

cv.createTrackbar('R','image',0,255,nothing)
cv.createTrackbar('G','image',0,255,nothing)
cv.createTrackbar('B','image',0,255,nothing)
switch = '0 : OFF \n1 : ON'
cv.createTrackbar(switch,'image',0,1,nothing)

while(1):
    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    r = cv.getTrackbarPos('R','image')
    g = cv.getTrackbarPos('G', 'image')
    b = cv.getTrackbarPos('B', 'image')
    s = cv.getTrackbarPos(switch,'image')
    if s == 0:
        img[:]=0
    else:
        img[:] = [b,g,r]

cv.destroyAllWindows()

#############################################################################
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

#############################################################################
#Arithmetic Operations
x = np.uint8([250])
y = np.uint8([10])
cv.add(x,y)  # [[255]]  250 + 10 = 260 > 255  -> 255
x+y          # 4        250 + 10 = 260 & 0XFF

#Image blending
#dst = a * img1 + b * img2 + c
img1 = cv.imread("/home/andy/1.jpeg")
img2 = cv.imread("/home/andy/2.jpeg")
#below operation runs ok only when img1 and img2 has the same shape
dst = cv.addWeighted(img1, 0, img2, 1, 0)

#Image Bitwise
img2 = cv.imread("/home/andy/logo.png")
#get row,col if the object pic
rows, cols, channels = img2.shape
#region of interesting
roi = img1[0:rows,0:cols]
#convert ot gray image and remove pixel by threshold value
img2gray  = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray,10,255,cv.THRESH_BINARY)
#mask of that pixel not needed
mask_inv = cv.bitwise_not(mask)

img1_bg = cv.bitwise_and(roi, roi, mask = mask_inv)
#those pixels not neede are set to zero by bitwise and operation
img2_fg = cv.bitwise_and(img2,img2,mask = mask)

dst=cv.add(img1_bg,img2_fg)
img1[0:rows,0:cols] = dst
#scale the image
img1 = cv.resize(img1,(0,0),fx=0.1,fy=0.1,interpolation=cv.INTER_NEAREST)

cv.imshow('Dst', img1)
cv.waitKey(0)
cv.destroyAllWindows()

#############################################################################

#measure the performance of code
e1 = cv.getTickCount()
print("Task running")
e2 = cv.getTickCount()
time = (e2 - e1)/cv.getTickFrequency()

#check optimized
cv.setUseOptimized(True)
img1 = cv.imread('/home/andy/logo.jpeg')
e1 = cv.getTickCount()
for i in range(5,49,20):
    img1 = cv.medianBlur(img1,i)
e2 = cv.getTickCount()
t = (e2 - e1)/cv.getTickFrequency()
print(t)

#check optimized
cv.setUseOptimized(False)
img2 = cv.imread('/home/andy/logo.jpeg')
e1 = cv.getTickCount()
for i in range(5,49,20):
    img2 = cv.medianBlur(img2,i)
e2 = cv.getTickCount()
t = (e2 - e1)/cv.getTickFrequency()
print(t)

#Use %timeit in python console
#use timeit in code
from timeit import timeit
def func(x,y):
    return x+y*2

t = timeit('func(5,6)','from __main__ import func',number=10000000)
print(t)

#Performance optimizatioin techniques
#try to implement the algorithm in a simple manner.
#. Avoid using loops in Python as far as possible, especially double/triple loops etc.
#  They are inherently slow.

#. Vectorize the algorithm/code to the maximum possible extent because Numpy and OpenCV
#  are optimized for vector operations.

#. Exploit the cache coherence.

#. Never make copies of array unless it is needed. Try to use views instead. Array
#  copying is a costly operation.

#############################################################################
# Changing Colorspaces
#Tips
flags = [i for i in dir(cv) if i.startswith('COLOR')]
print(flags)
#change  a color value.
green = np.uint8([[[0,255,0]]])  #an image with only one pixel
hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)

#for HSV , different software use diffenect scales, here, h:[0,179],S,V[0,255]

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open, exit")
    exit()
while (1):
    _, frame = cap.read()
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    lower_blue = np.array([110,50, 50])
    upper_blue = np.array([130,255,255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    res = cv.bitwise_and(frame, frame, mask = mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    #How to remove the noises?TBD
    k = cv.waitKey(5) & 0xFF
    if k == ord('q'):
        break
cv.destroyAllWindows()
#############################################################################
#Image transfomations
# cv.warpAffine and cv.warpPerspective()
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
# what do below points mean? refer to:
# https://jingyan.baidu.com/article/a17d5285c128cc8098c8f2f2.html(base) 
pts1 = np.float32([[56, 65], [368, 52], [28, 387],[389,390]])
pts2 = np.float32([[ 0,  0], [300,  0], [ 0, 300],[300,300]])
M = cv.getPerspectiveTransform(pts1, pts2)
dst = cv.warpPerspective(res, M, (300, 300))
showImage('input',res)
showImage('output',dst)
#############################################################################
#Threshold
from matplotlib import pyplot as plt
#Imgae threashold

img = cv.imread('/home/andy/2.jpeg',0)
# img = cv.medianBlur(img,5)
ret, th1 = cv.threshold(img,80,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,    cv.THRESH_BINARY,25,2)
# th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,    cv.THRESH_BINARY,15,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,25,2)
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
ret, thresh3 = cv.threshold(img,72,255,cv.THRESH_TRUNC)
ret, thresh4 = cv.threshold(img,72,96,cv.THRESH_TOZERO)
ret, thresh5 = cv.threshold(img,72,96,cv.THRESH_TOZERO_INV)

titles = ['Org','Bin','IBin','Trc','TZero','ITZero']
imgs = [img,thresh1,thresh2,thresh3,thresh4,thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(imgs[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
#############################################################################
#OTSU!!!
import matplotlib.pyplot as plt

img =  cv.imread('/home/andy/noisey.png',0)#what does 0 means?
#global threasholding
ret1, th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
#Otu's thresholding
ret2, th2 = cv.threshold(img,  0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
#Otu's thresholding after gaussioin filtering
blur = cv. GaussianBlur(img,(5,5),0)
ret2, th3 = cv.threshold(blur,  0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
imgs = [img, 0, th1,
        img, 0, th2,
       blur, 0, th3]
titles = ['orig','histogram','Global Thresholding(v=127)',
          'orig','histogram','OTSUs Thresholding',
          'blur','histogram','OTSUs Thresholding']
for i in range(3):
    plt.subplot(3,3,i*3+1), plt.imshow(imgs[i*3],'gray')
    plt.title(titles[i*3]),plt.xticks([]),plt.yticks([])

    plt.subplot(3,3,i*3+2),plt.hist(imgs[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])

    plt.subplot(3, 3, i * 3 + 3), plt.imshow(imgs[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

plt.show()
#another example use OTSU
img = cv.imread("/home/andy/1.jpeg", -1)
img = cv.resize(img,(0,0),fx=0.1,fy=0.1,interpolation=cv.INTER_CUBIC)
img = cv.GaussianBlur(img, (3,3),0)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

retval, dst = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
cv.imshow("src", img)
cv.imshow("gray", gray)
cv.imshow("dst", dst)
cv.waitKey(0)
#############################################################################
#SMOOTH, LPF low pass filter  HPF high pass filter, LPF helps in removing noise
#HPF helps in finding edges
import matplotlib.pyplot as plt
img = cv. imread("/home/andy/opencvlogo.png")
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
#############################################################################
#Morphological transform
#物体形态处理，侵蚀　膨胀，梯度，黑帽，白帽,...
img = cv .imread('/home/andy/j.png',0)
kernel = np.ones((5,5),np.uint8)
erosion  =  cv.erode(img,kernel,iterations = 1)
dilation = cv.dilate(img,kernel,iterations = 1)
opening  = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
closing  = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
tophat   = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

imgs = [img,erosion,dilation,opening,closing,gradient,tophat,blackhat]
title = ["orig","erosion","dilation","opening","closing","gradient","tophat","blackhat"]
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(imgs[i])
    plt.title(title[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
#Technque cv.getStructuringElement(cv2.MORPH_RECT,(5,5))

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
#############################################################################
#Canny edge detection is a popular edge detection algorithm
#Noise reduction
#Finding intensiy gradient of the Image
#Non-maximum suppression
#Hysteresis threshloding
# 应用高斯滤波来平滑图像，目的是去除噪声
# 找寻图像的强度梯度（intensity gradients）
# 应用非最大抑制（non-maximum suppression）技术来消除边误检（本来不是但检测出来是）
# 应用双阈值的方法来决定可能的（潜在的）边界
# 利用滞后技术来跟踪边界
img = cv.imread('/home/andy/cannytest1.png',0)
edges = cv.Canny(img,0,255)

imgs = [img,edges]
title = ["orig",'edges']
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(imgs[i],cmap='gray')
    plt.title(title[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
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
#############################################################################
#contour
img = cv .imread('/home/andy/apple.png')
imggray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret,thresh = cv.threshold(imggray,64,255,0)
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
#############################################################################
#Histogram
# an overall idea about the intensity distribution of an imag
img = cv .imread('/home/andy/apple1.png',0)
# hist = cv.calcHist([img],[0],None,[256],[0,256])
#np
# hist, bin = np.histogram(img.ravel(),256,[0,256])
#show
# plt.hist(img.ravel(),256,[0,256])
# plt.show()

# color = ('b','g','r')
# for i, col in enumerate(color):
#     histr = cv.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color=col)
#     # plt.xlim([0,256])
# plt.show()
#Find histograms of some regions of an image
mask = np. zeros(img.shape[:2],np.uint8)
mask [0:128, 0:256]=255
masked_img = cv.bitwise_and(img, img, mask=mask)

hist_full = cv.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv.calcHist([img],[0],mask,[256],[0,256])

plt.subplot(221),plt.imshow(img,'gray')
plt.subplot(222),plt.imshow(mask,'gray')
plt.subplot(223),plt.imshow(masked_img,'gray')
plt.subplot(224),plt.plot(hist_full),plt.plot(hist_mask)
plt.xlim([0,256])
plt.show()
#Histogram
# an overall idea about the intensity distribution of an imag
img = cv .imread('/home/andy/apple1.png',0)
hist, bin = np.histogram(img.flatten(),256,[0,256])
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
#############################################################################
#2D Histogram,Normally it is used for finding color histograms where two
# features are Hue & Saturation values of every pixel
img = cv.imread('/home/andy/1.jpeg')
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
hist = cv.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
plt.imshow(hist,interpolation='nearest')
plt.show()
#numpy
h = hsv[:,:,0]
s = hsv[:,:,1]
hist,xbins,ybins = np.histogram2d(h.ravel(),s.ravel(),[180,256],[[0,180],[0,256]])
#############################################################################
# histogram backprojection.
#the output image will have our object of interest in more white compared to
#two ways: using numpy , using cv

roi = cv.imread('/home/andy/testroi.png')
hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)

target = cv.imread('/home/andy/test.jpg')
hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)
# calculating object histogram
roihist = cv.calcHist([hsv],[0, 1], None, [128, 256], [0, 180, 0, 256] )
# normalize histogram and apply backprojection
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)

dst = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)

# Now convolute with circular disc
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)) #

cv.filter2D(dst,-1,disc,dst)
# threshold and binary AND
ret,thresh = cv.threshold(dst,50,255,0)
thresh = cv.merge((thresh,thresh,thresh))
res = cv.bitwise_and(target,thresh)
res = np.hstack((target,thresh, res))
cv.imshow('res',res)
cv.waitKey(0)
cv.destroyAllWindows()
#############################################################################
# For images, 2D Discrete Fourier Transform (DFT) is used to find the
# frequency domain. A fast algorithm called Fast Fourier Transform (FFT) is
# used for calculation of DFT.
# x(t)=Asin(2πft)
# Numpy has an FFT package to do this. np.fft.fft2()
img = cv.imread('/home/andy/me.jpeg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
magnitude_spectrum = magnitude_spectrum/256  #why this step is need to imshow

# plt.subplot(121), plt.imshow(img,               cmap='gray')
# plt.subplot(122), plt.imshow(magnitude_spectrum,cmap='gray')
# plt.show()

rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)
xrang = 250
fshift[crow-xrang:crow+xrang, ccol-xrang:ccol+xrang] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
# plt.subplot(221), plt.imshow(img,cmap='gray'),      plt.xticks([]), plt.yticks([])
# plt.subplot(222), plt.imshow(img_back,cmap='gray'), plt.xticks([]), plt.yticks([])
# plt.subplot(223), plt.imshow(img_back),             plt.xticks([]), plt.yticks([])
# plt.show()

#cv method
# magnitude:震级；巨大；重大；重要性　spectrum:频谱；范围；领域；序列
dft = cv.dft(np.float32(img),flags= cv.DFT_COMPLEX_OUTPUT)
#fft得到的像谱默认不是按照中心对称的（快速傅里叶变换的原因），一般需要用fftshift方法使得其按中
# 心对称，这样的话当我们ifft 时，得到的数据就会和之前实际的不一样了，所以还需加ifftshift 来还原
dft_shift = np.fft.fftshift(dft)
idft      = np.fft.ifftshift(dft_shift)
dft_shift2 = np.fft.fftshift(idft)
#频谱，１３　相同
magnitude_spectrum1 = 20 * np.log(cv.magnitude(dft[:,:,0],            dft[:,:,1]))
magnitude_spectrum2 = 20 * np.log(cv.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
magnitude_spectrum3 = 20 * np.log(cv.magnitude(dft_shift2[:,:,0],dft_shift2[:,:,1]))

plt.subplot(141), plt.imshow(img,                cmap='gray')
plt.subplot(142), plt.imshow(magnitude_spectrum1,cmap='gray')
plt.subplot(143), plt.imshow(magnitude_spectrum2,cmap='gray')
plt.subplot(144), plt.imshow(magnitude_spectrum3,cmap='gray')
plt.show()


rows, cols = img.shape
print(img.shape)
crow, ccol = int(rows/2),int(cols/2)
mask = np.zeros((rows,cols,2),np.uint8)
x=300
mask[crow-x:crow+x,ccol-x:ccol+x]= 1
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])
# plt.subplot(121), plt.imshow(img,     cmap='gray')
# plt.subplot(122), plt.imshow(img_back,cmap='gray')
# plt.show()

#Different filter
# Why below arraies represent different filters?
mean_filter = np.ones((3,3))

x = cv.getGaussianKernel(5,10)
#T 对矩阵的一个转置，旋转９0度
gaussian = x * x.T

scharr = np.array([[ -3, 0, 3],
                   [-10, 0,10],
                   [ -3, 0, 3]])

sobel_x = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])

sobel_y = np.array([[-1,-2,-1],
                    [ 0, 0, 0],
                    [ 1, 2, 1]])

laplacian = np.array([[0, 1,0],
                      [1,-4,1],
                      [0, 1,0]])

filters = [mean_filter,gaussian,laplacian,sobel_x,sobel_y,scharr]
filters_name =['mean_filter','gaussian','laplacian','sobel_x','sobel_y','scharr']

fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log(np.abs(z)+1) for z in fft_shift]

for i in range(6):
    # print(magnitude_spectrum[i].shape)
    plt.subplot(2,3,i+1), plt.imshow(mag_spectrum[i],cmap='gray')
    plt.title(filters_name[i]),plt.xticks([]),plt.yticks([])
plt.show()
#############################################################################
#Template matching
img = cv.imread('/home/andy/me.jpeg',0)
img2 = img.copy()

template = cv.imread('/home/andy/met.png',0)
w,h = template.shape[::-1]

# TM_CCORR：相关性匹配　　　　　TM_CCORR_NORMED:标准相关性匹配
# TM_CCOEFF:相关性系数匹配   　TM_CCOEFF_NORMED：标准相关性系数匹配
# TM_SQDIFF:平方差匹配 　　　　TM_SQDIFF_NORMED:标准平方差匹配
methods = ['cv.TM_CCOEFF',        'cv.TM_CCOEFF_NORMED',  'cv.TM_CCORR',
           'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF',         'cv.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    res = cv.matchTemplate(img,template,method)
    min_val,max_al,min_loc,max_loc = cv.minMaxLoc(res)
    print(min_val)
    print(max_al)
    print(min_loc)
    print(max_loc)

    if method in [cv.TM_SQDIFF,cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0]+w, top_left[1]+h)

    cv.rectangle(img, top_left,bottom_right,255,2)

    # plt.subplot(221),plt.imshow(res,cmap='gray')
    # plt.title('Matching result'), plt.xticks([]),plt.yticks([])
    # plt.subplot(222),plt.imshow(img,cmap='gray')
    # plt.title('detected point'), plt.xticks([]), plt.yticks([])
    # plt.subplot(223), plt.imshow(template, cmap='gray')
    # plt.title('template'), plt.xticks([]), plt.yticks([])
    # plt.suptitle(meth)
    # plt.show()
# match more than one objects
img_rgb = cv.imread('/home/andy/mario.jpeg')
img_gray = cv.cvtColor(img_rgb,cv.COLOR_BGR2GRAY)
template = cv.imread('/home/andy/mt.png')
template = cv.cvtColor(template,cv.COLOR_BGR2GRAY)

w,h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold =0.9
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb,pt,(pt[0]+w,pt[1]+h),(0,255,0),1)

plt.imshow(img_rgb)
plt.show()

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
#############################################################################
#HoughCircles
img = cv.imread('/home/andy/opencvlogo.png',0)
img = cv.medianBlur(img,5)
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),1)
    # draw the center of the circleq
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
cv.imshow('detected circles',cimg)
cv.waitKey(0)
cv.destroyAllWindows()
#############################################################################
#Watershed 分水岭，重大变化的时间或周期
# https://blog.csdn.net/dcrmg/article/details/52498440
img = cv.imread('/home/andy/coin.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

#noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel,iterations=2)
#sure background
sure_bg = cv.dilate(opening,kernel,iterations=3)
# distanceTransform()用于计算图像中每一个非零点像素与其最近的零点像素之间的距离，输出
# 的是保存每一个非零点与最近零点的距离信息；可以根据距离变换的这个性质，经过简单的运算，
# 用于细化字符的轮廓和查找物体质心
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret,sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)

ret, markers = cv.connectedComponents(sure_fg)
markers = markers +1
markers[unknown==255] = 0
markers = cv.watershed(img,markers)
img[markers == -1]=[255,255,0]

plt.subplot(221), plt.imshow(markers),plt.xticks([]),plt.yticks([])
plt.subplot(222), plt.imshow(sure_bg),plt.xticks([]),plt.yticks([])
plt.subplot(223), plt.imshow(sure_fg),plt.xticks([]),plt.yticks([])
plt.subplot(224), plt.imshow(img),plt.xticks([]),plt.yticks([])
plt.show()
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

#############################################################################
#Harris corner detection
img = cv.imread('/home/andy/pics/chess.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)
#dilate 膨胀图像，参见变形处理
dst = cv.dilate(dst,None)
img[dst > 0.09 * dst.max() ] = [0,255,255]
cv.imshow('dst',img)
cv.waitKey(0)
cv.destroyAllWindows()

img = cv.imread('/home/andy/pics/sumiao.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)
#dilate 膨胀图像，参见变形处理
dst = cv.dilate(dst,None)
ret,dst = cv.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
ret,labels,stats,centroids = cv.connectedComponentsWithStats(dst)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,100,0.001)
corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
res = np.hstack((centroids,corners))
#最接近0的整数，12.9 -> 12  || -8.1  -> -8
res = np.int0(res)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]]=[0,255,0]
cv.imwrite('/home/andy/pics/1015.png',img)

#############################################################################
#Shi-Tomasi Corner detector and good features to track
img = cv.imread('/home/andy/pics/chess.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
corners =  cv.goodFeaturesToTrack(gray,20,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y= i.ravel()
    cv.circle(img,(x,y),4,255,-1)

plt.imshow(img)
plt.show()
#############################################################################
#SIFT (Scale-Invariant Feature Transform)尺度不变特征变换
img = cv.imread('/home/andy/pics/1.jpeg')
fast =  cv.FastFeatureDetector_create()
kp = fast.detect(img,None)
img2 =  cv.drawKeypoints(img,kp,None,color=(0,0,255))
fast.setNonmaxSuppression(5) #镇压; 压制; 抑制;
kp = fast.detect(img,None)
img3 =  cv.drawKeypoints(img,kp,None,color=(0,255,0))

plt.subplot(121), plt.imshow(img2)
plt.subplot(122), plt.imshow(img3)
plt.show()
#############################################################################
#Meanshift and Camshift
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open, exit")
    exit()

# cap.set(cv.CAP_PROP_FRAME_WIDTH, 512)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 384)

ret, frame = cap.read()
frame = cv.flip(frame, 1)

# r,h,c,w = 0,115,245,139
r,h,c,w = 245,139,0,115
track_window = (c,r,w, h)

roi = frame[r:r+h,c:c+w]
# cv.imshow('roi',roi)
# cv.waitKey(19000)
# cv.destroyAllWindows()

hsv_roi = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi,np.array((0.,60.,32.)),np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_EPS,10,1)

while (1):
    ret, frame = cap.read()
    if ret == True:
        frame = cv.flip(frame, 1)
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        ret, track_window = cv.meanShift(dst,track_window,term_crit)
        ret, track_window = cv.CamShift(dst, track_window, term_crit)

        x,y,w,h = track_window
        img2 =  cv.rectangle(frame,(x,y),(x+w,y+h),255,2)
        # cv.imshow('frame', img2)

        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame, [pts], True, 255, 2)
        cv.imshow('img2', img2)

        k=cv.waitKey(24) & 0xff
        if k==27:
            break;

cv.destroyAllWindows()
cap.release()