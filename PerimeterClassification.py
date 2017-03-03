import numpy as np
import NN as network
import sys
import random


if __name__ == '__main__':

    alpha = 0.1
    num_iter = 10000
    n_in,n_hidden,n_out = 2,7,1
    train_set = np.loadtxt("datos_P2_EM2017_N2000.txt")
    x,y = network.extractData(train_set,n_out)
    x1 = network.normalize(x)
    print x1
    # random.seed(1)
    nn = network.createNeuralNetwork(n_in,n_hidden,n_out)
    error = network.backPropagation(nn,x1,y,alpha,num_iter)

    # for i in range(len(train_set)):
    #     print str(x[i]) + " " + str(round(network.test(nn,x[i])[0])) + " " + str(round(network.test(nn,x[i])[0]) == y[i]) + " " + str(i+1) + "

    # -------------- TEST--------------------
    good_predictions, bad_predictions, fp ,fn = 0, 0, 0, 0
    print "\nID     Instance        Prediction Result  "
    for i in range(len(x1)):
        out = network.test(nn, x1[i])
        result = round(out[0])
        correct = result == y[i]
        if (correct):
            good_predictions += 1
        else:
            bad_predictions += 1
            if (result == 1) : fp += 1
            else : fn += 1
        print str(i + 1) + " " + str(x[i]) + " " + str(result) + " " + str(correct)
    print "------------------ Results ---------------------"
    print ("Iterations = %d     Learning Rate = %.3f") % (num_iter, alpha)
    print ("Error Value = %.4f") % (error)
    print ("Correctly Predicted = %d     Poorly Predicted = %d") % (good_predictions, bad_predictions)
    print ("False Positives = %d     False Negatives = %d") % (fp, fn)