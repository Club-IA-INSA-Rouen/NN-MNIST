import gzip
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def visualize(inputNumber):
    with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)
    train_x, train_y = train_set

    plt.imshow(train_x[inputNumber].reshape((28, 28)), cmap=cm.Greys_r)
    plt.show()
