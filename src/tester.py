import random
import numpy as np
import matplotlib.pyplot as plt
from network import Network as NN
from visualizing_data import visualize
from importing_data import *
# now serious business with a real sample!

randrow = random.randint(0,50000)
train, valid, test = clean_data()
gud_net = NN([784, 16, 10])

batches_size = 20
nb_epochs = 70
eta = 10.
epochs, succ_rates = gud_net.stochGrad(batches_size,nb_epochs,train,eta,test,decrease=True)

# Uncomment if you feel like checkin it up !
x,y = train[randrow]
print "Should be : " + str(from_unit(y))
print "Decision is : " + str(gud_net.prevision(x))
visualize(x)

if len(succ_rates) > 0:
    plt.plot(epochs, succ_rates)
    plt.ylabel('Success rate')
    plt.show()
