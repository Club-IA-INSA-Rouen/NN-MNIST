import cv2
import numpy as np
from visualizing_data import visualize

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing :
            cv2.circle(img,(x,y),circleSize,white,-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def pixelToValue(pixel):
    return 1 if pixel.any() != 0 else 0

def imgToReduced(img, imSize, mult):
    img = np.array([[pixelToValue(pix) for pix in row] for row in img])
    sub_img = np.array([[img[x*mult:((x+1)*mult) , y*mult:((y+1)*mult)].mean()\
        for x in range(imSize)] for y in range(imSize)])
    return sub_img
