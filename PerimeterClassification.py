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
    train_set = np.loadtxt("datos_P2_EM2017_N500.txt")
    x,y = extractData(train_set)
    # random.seed(1)
    nn = network.createNeuralNetwork(len(train_set[0])-1,2,1)
    print nn
    network.backPropagation(nn,train_set,0.01,2000)
    print nn
    print network.forwardPropagate(nn,x[75])