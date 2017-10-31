import numpy as np


class Network(object):
    # Neural Network with propagation and backpropagation
    def __init__(self, layerSizes):
        """ let's be careful with matrices sizes !
        nrow = nb neurons in the next layer
            --> goes from first ofhidden to output
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

    def StochGrad(arg):
        pass

    def display(self):
        for w,b in zip(self.weights, self.biases):
            print w
            print b

# Real Functions (activation)
def sig(x): # x is a vector from numpy; operation done elementwise
    return 1./(1.+np.exp(-x)) # np.exp() applies it elementwise
