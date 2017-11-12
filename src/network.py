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

    # same thing but for backprop so we output values of activations and zs
    def propagation_save(self, input):
        a = [input] # list of activations, a0 = x = input
        z = [] # list of aw+b (before applying activation function)
        for w,b in zip(self.weights, self.biases):
            z.append(np.dot(w,a[-1]) + b)
            a.append(sig(z[-1]))
        return a,z


    def prevision(self, input):
        return np.argmax(self.propagation(input))

    def stochGrad(self, batches_size, nb_epochs, training_data, learning_rate,
        test_data=None, decrease=False):
        """ Now for the learning.
        Said stochastic since we're only going for steps on minibatches
        but it ends up being nearly as good and much faster.
        Repeating this over a set number of epoch.
        """
        # as in the book, let's prevent too much accesses
        epochs = []
        succ_rates = []
        every = 2
        nb_training = len(training_data)
        if decrease:
            common_diff = learning_rate/(nb_epochs+5.)
        if test_data:
            nb_test = len(test_data)
        for ep in xrange(nb_epochs): # now, for each epoch :
            # shuffle data, create batches, compute and apply modifications
            random.shuffle(training_data)
            batches = [training_data[k:k+batches_size]
                        for k in xrange(0,nb_training,batches_size)]
            # let's train the nn for each batch !
            for batch in batches:
                self.learnBatch(batch, learning_rate)
            if decrease:
                learning_rate -= common_diff
            if (ep%every==0) and test_data: # if you want, print progress
                successes = self.nb_success(test_data)
                success_rate = float(successes)/nb_test
                epochs.append(ep)
                succ_rates.append(success_rate)
                print("Epoch " + str(ep) + " : "
                + str(successes) + "/" + str(nb_test)
                + " ~ " + str(success_rate))
        return epochs, succ_rates

    def learnBatch(self, batch, learning_rate):
        """ Learning from one batch of data
        We compute the gradient delta_grad_b delta_grad_w,
        for each pair (x,y) of the batch;
        Remembering that grad_ = sum of delta_grad_
        We then update weights and biases by adding (learn_rate/sizebatch)*grad_
        """
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        for x,y in batch:
            delta_grad_w, delta_grad_b = self.backpropagation(x,y)
            # those are delta values for each layer. Let's add them to grad_ :
            grad_w = [gw+dgw for gw,dgw in zip(grad_w,delta_grad_w)]
            grad_b = [gb+dgb for gb,dgb in zip(grad_b,delta_grad_b)]
        # now let's slide down the hill of cost !
        ratio_learn = (learning_rate/len(batch))
        self.weights = [w-ratio_learn*gw for w,gw in zip(self.weights,grad_w)]
        self.biases = [b-ratio_learn*gb for b,gb in zip(self.biases,grad_b)]

    def backpropagation(self, x,y):
        """ x: input, y: output; let's compute the gradient of C(w,b) for this
        pair and the participation of each and every w(l,i,j) and b(l,i)
        """
        delta_grad_w = [np.zeros(w.shape) for w in self.weights]
        delta_grad_b = [np.zeros(b.shape) for b in self.biases]
        a,z = self.propagation_save(x) # activations and zs
        # Initialisation of backprop :
        ## we have to test this ! delta = cost_der * a[-1]*(1-a[-1])
        delta = cost_der(a[-1], y)*sig_prime(z[-1])
        delta_grad_w[-1] = np.dot(delta,a[-2].T) # just check the definition ^^
        delta_grad_b[-1] = delta
        # looping backprop
        for l in xrange(2, self.nLayers): # we already initialized,
            # and b and w have one less layer than total number (bcz of input)
            delta = np.dot(self.weights[-l+1].T, delta) * sig_prime(z[-l])
            delta_grad_w[-l] = np.dot(delta,a[-l-1].T) # just check the definition ^^
            delta_grad_b[-l] = delta

        return delta_grad_w, delta_grad_b

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

def sig_prime(x):
    sx = sig(x)
    return sx*(1-sx)

def cost_der(aL,y):
    return aL-y
