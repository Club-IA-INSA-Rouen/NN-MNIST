import numpy as np
import random

class Network(object):
    # Neural Network with propagation and backpropagation
    def __init__(self, layerSizes):
        """ let's be careful with matrices sizes !
        nrow = nb neurons in the next layer
            --> goes from first of hidden to output
        ncol = nb neurons in current layer
            --> goes from input to last of hidden
        """
        self.layerSizes = layerSizes # list of nb of neurons per layer
        self.nLayers = len(layerSizes)
        self.biases = [np.random.randn(x,1) for x in layerSizes[1:]]
        self.weights = [np.random.randn(x,y)
            for x,y in zip(layerSizes[1:],layerSizes[:-1])]

    # propagation: keep doing Input(i+1) = sigmoid(Weight.Input(i) + Bias)
    def propagation(self,input): # input is propagated until it becomes output
        for w,b in zip(self.weights, self.biases): # note that they're same size
            input = sig(np.dot(w, input) + b)
        return input

    def prevision(self, input):
        return np.argmax(self.propagation(input))

    def stochGrad(self, batches_size, nb_epochs, training_data, test_data=None):
        """ Now for the learning.
        Said stochastic since we're only going for steps on minibatches
        but it ends up being nearly as good and much faster.
        Repeating this over a set number of epoch.
        """
        # as in the book, let's prevent too much accesses
        nb_training = len(training_data)
        if test_data:
            nb_test = len(test_data)
        for ep in xrange(nb_epochs): # now, for each epoch :
            # shuffle data, create batches, compute and apply modifications
            random.shuffle(training_data)
            batches = [training_data[k:k+batches_size]
                        for k in xrange(0,nb_training,batches_size)]
            # let's train the nn for each batch !
            for batch in batches:
                self.learnBatch(batch)
            if test_data: # if you want, print progress
                successes = self.nb_success(test_data)
                print("Epoch " + str(ep) + " : "
                + str(successes) + "/" + str(nb_test)
                + " ~ " + str(float(successes)/nb_test))

    def learnBatch(self, batch):
        """ Learning from one set of data
        """
        pass

    def gradient(self, arg):
        pass

    def nb_success(self, test): # returns nb of good previsions over the test data
        results_couples = [(self.prevision(x), y)
                        for (x, y) in test]
        # now let's count the good ones - remembering that int(bool) = 0 or 1
        return sum(int(x == y) for (x, y) in results_couples)


    def display(self):
        for w,b in zip(self.weights, self.biases):
            print w
            print b

# Real Functions (activation)
def sig(x): # x is a vector from numpy; operation done elementwise
    return 1./(1.+np.exp(-x)) # np.exp() applies it elementwise
