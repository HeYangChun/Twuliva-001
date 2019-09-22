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
