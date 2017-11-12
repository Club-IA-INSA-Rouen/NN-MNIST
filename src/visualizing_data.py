import gzip
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def visualize(pixels):
    plt.imshow(pixels.reshape((28, 28)), cmap=cm.Greys_r)
    plt.show()
