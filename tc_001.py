import numpy as np
import cv2 as cv

#load image
img = cv.imread("PIC URL",0)

#show
cv.imshow("WINDOW TITLE",img)
cv.waitKey(0)
cv.destroyAllWindows()

#save image
cv.imwrite("FILENAME",img)

#using matplotlib
from matplotlib import pyplot as plt
plt.imshow(img,cmap='gray',interploation='bicubic')
plt.xticks([]), plt.yticks([]) #hide tick values on X and Y axis
plt.show()

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
