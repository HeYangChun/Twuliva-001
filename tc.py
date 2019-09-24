import numpy as np
import cv2 as cv

#Tips
flags = [i for i in dir(cv) if i.startswith('COLOR')]
print(flags)
#for HSV , different software use diffenect scales, here, h:[0,179],S,V[0,255]

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open, exit")
    exit()
while (1):
    _, frame = cap.read()
    # frame = cv.flip(frame,2)
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    lower_blue = np.array([110,50, 50])
    upper_blue = np.array([130,255,255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    res = cv.bitwise_and(frame, frame, mask = mask)
    cv.imshow('frame',frame)
    # cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == ord('q'):
        break
cv.destroyAllWindows()