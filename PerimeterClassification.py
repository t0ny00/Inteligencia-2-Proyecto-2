import numpy as np
import NN as network
import sys
import random
import matplotlib.pyplot as plt


def costFunctionPlot(data, numberIterations, alpha):
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y)

    plt.title(r'Convergence Curve for $\alpha=$' + str(alpha) + ' and ' + str(numberIterations) + ' iterations')
    plt.ylabel('Cost Function Value')
    plt.xlabel('Number of Iterations')
    plt.show()

if __name__ == '__main__':

    alpha = float(sys.argv[1])
    file_name = "datos_P2_EM2017_N2000.txt"
    num_iter = 8000
    n_in,n_hidden,n_out = 2,int(sys.argv[2]),1
    train_set = np.loadtxt(file_name)
    x,y = network.extractData(train_set,n_out)
    x1 = network.normalize(x)
    # random.seed(1)
    nn = network.createNeuralNetwork(n_in,n_hidden,n_out)
    error,log = network.backPropagation(nn,x1,y,alpha,num_iter)


    # -------------- TEST--------------------
    test_set = np.loadtxt("generated_10000.txt")
    x, y = network.extractData(test_set, n_out)
    x1 = network.normalize(x)
    good_predictions, bad_predictions, fp ,fn = 0, 0, 0, 0
    print "\nID     Instance        Prediction Result  "
    for i in range(len(x1)):
        out = network.predict(nn, x1[i])
        result = round(out[0])
        correct = result == y[i]
        if (correct):
            good_predictions += 1
        else:
            bad_predictions += 1
            if (result == 1) : fp += 1
            else : fn += 1
        # print str(i + 1) + " " + str(x[i]) + " " + str(result) + " " + str(correct)
    print "------------------ Results ---------------------"
    print ("File = %s Iterations = %d     Learning Rate = %.3f  Hidden Layer = %d") % (file_name,num_iter, alpha,n_hidden)
    print ("Error Value = %.4f") % (error)
    print ("Correctly Predicted = %d     Poorly Predicted = %d") % (good_predictions, bad_predictions)
    print ("False Positives = %d     False Negatives = %d") % (fp, fn)
    costFunctionPlot(log,num_iter,alpha)