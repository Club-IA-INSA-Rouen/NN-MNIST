import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from network import Network as NN
from visualizing_data import visualize
from importing_data import *
#from drawer import *
np.set_printoptions(suppress=True)

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

# now serious business with a real sample!

hidden = [int(a) for a in argv[1:]] if len(argv) > 1 else [100]
nn_skel = [784]+hidden+[10]
print "Hidden : " + str(nn_skel[1:-1])
gud_net = NN(nn_skel)

train, valid, test = clean_data()
batches_size = 10
nb_epochs = 35
eta = 0.2
reg_param = 5.
epochs, succ_rates = gud_net.stochGrad(batches_size,\
    nb_epochs,train,eta,test,decrease=False,reg_param=reg_param)

# plotting performance over time
plot_result = False
if ((len(succ_rates) > 0) and plot_result) :
    plt.plot(epochs, succ_rates)
    plt.ylabel('Success rate')
    plt.show()

# Uncomment if you feel like checkin it up !
nbVisu = 10
for testvisu in range(nbVisu):
    randrow = random.randint(0,10000)
    x,y = test[randrow]
    print "Probability vector : "
    print gud_net.propagation(x)
    print "Decision is : " + str(gud_net.prevision(x))
    visualize(x)

# now let's wave artistic
drawing = False # true if mouse is pressed
ix,iy = -1,-1
white = (255,255,255)
circleSize = 15
imSize = 28
mult = 15
bigSize = mult*imSize
nb_drawn = 1000
# the for loop for each drawing
for drawn in range(nb_drawn):
    img = np.zeros((bigSize,bigSize,3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k in (27,32,10):
            break
    cv2.destroyAllWindows()
    final_img = np.transpose(imgToReduced(img,imSize,mult))

    final_data = np.reshape(final_img,(784,1))

    print "Probability vector : "
    print gud_net.propagation(final_data)
    print "Decision is : " + str(gud_net.prevision(final_data))

    visualize(final_img)
