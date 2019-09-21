import numpy as np
import cv2 as cv

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
