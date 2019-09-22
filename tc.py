import numpy as np
import cv2 as cv

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
