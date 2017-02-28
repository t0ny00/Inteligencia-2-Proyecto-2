import numpy as np
import NN as network
import sys
import random

def extractData(data):
    numberColumns = data.shape[1]
    y = data[:, numberColumns - 1]
    x = data[:, 0:numberColumns - 1]
    return x,np.reshape(y,(len(y),1))


if __name__ == '__main__':

    alpha = 0.7
    num_iter = 1000
    n_in,n_hidden,n_out = 2,5,1
    train_set = np.loadtxt("generated_100.txt")
    x,y = extractData(train_set)
    # random.seed(1)
    nn = network.createNeuralNetwork(n_in,n_hidden,n_out)
    print nn
    network.backPropagation(nn,train_set,alpha,num_iter)
    print nn
    print network.forwardPropagate(nn,x[2])