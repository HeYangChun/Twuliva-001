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


#############################################################################
