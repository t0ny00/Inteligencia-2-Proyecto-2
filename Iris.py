import numpy as np
import NN as network
import sys
import random




if __name__ == '__main__':

    alpha = 0.1
    num_iter = 10000
    n_in,n_hidden,n_out = 4,10,3
    train_set = np.loadtxt("iris_3classes.txt",delimiter=",")
    # print train_set
    x,y = network.extractData(train_set,n_out)
    # random.seed(1)
    nn = network.createNeuralNetwork(n_in,n_hidden,n_out)
    print nn
    # print nn
    network.backPropagation(nn,x,y,alpha,num_iter)
    print nn
    for i in range(len(train_set)):
        result = network.test(nn,x[i])
        if (np.argmax(result) == 0) : test = [1,0,0]
        elif (np.argmax(result) == 1) : test = [0,1,0]
        else: test = [0,0,1]
        print str(i+1) + " " + str(x[i]) + " " + str(test) + " " + str(test==y[i])