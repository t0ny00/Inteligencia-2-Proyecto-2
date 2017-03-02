import numpy as np
import NN as network
import sys
import random



if __name__ == '__main__':

    alpha = 0.1
    num_iter = 500
    n_in,n_hidden,n_out = 2,10,1
    train_set = np.loadtxt("generated_1000.txt")
    x,y = network.extractData(train_set,n_out)
    # random.seed(1)
    nn = network.createNeuralNetwork(n_in,n_hidden,n_out)
    print nn
    network.backPropagation(nn,x,y,alpha,num_iter)
    print nn
    for i in range(len(train_set)):
        print str(x[i]) + " " + str(round(network.test(nn,x[i])[0])) + " " + str(round(network.test(nn,x[i])[0]) == y[i]) + " " + str(i+1) + ""