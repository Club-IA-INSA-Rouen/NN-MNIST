import gzip
import cPickle

def importing_data():
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close
    return (train_set, valid_set, test_set)

def unit_vector(i): # turning the output into something nice :
    # returns unit vector e length 10 where ej = (i==j) ? 1:0
    e = np.zeros((10,1))
    e[i] = 1.
    return e

def clean_data():
    train, valid, test = importing_data()
    """ each of those is a list of pairs of data, where:
    x[0] is the input, 28*28=784 shades of grey in [0,1]
    x[1] is the output, the raw integer value in [[0,9]]
    we want to turn those into numpy matrices : """
    # output y as matrix means a unit vector as definined above
    train_input = [np.reshape(x,(784,1)) for x in train[0]]
    train_output = [unit_vector(y) for y in train[1]]
    clean_train = zip(train_input, train_output)
    # we don't need the modification on y since it won't serve to train the nn
    valid_input = [np.reshape(x,(784,1)) for x in valid[0]]
    clean_valid = zip(valid_input, valid[1])

    test_input = [np.reshape(x,(784,1)) for x in test[0]]
    clean_test = zip(test_input, test[1])
    return clean_train, clean_valid, clean_test
