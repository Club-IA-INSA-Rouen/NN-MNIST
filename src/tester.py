import numpy as np
from network import Network as NN
from visualizing_data import visualize
from importing_data import *
# now serious business with a real sample!

row = 42

train, valid, test = clean_data()
nub_net = NN([784, 16, 10])

x,y = train[row]
print "Should be : " + str(from_unit(y))
print "Decision is : " + str(nub_net.prevision(x))

# Uncomment if you feel like checkin it up !
visualize(row)
