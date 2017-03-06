# coding=utf-8
import numpy as np
import NN as network
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Universidad Simón Bolívar
# Inteligencia Artificial II
# Prof. Ivette Carolina Martínez
# Alumnos:
# Carlos Martinez 11-10584
# Yerson Roa 11-10876
# Antonio Scaramazza 11-10957

# Renders the the cost function plot
def costFunctionPlot(data, numberIterations, alpha):
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y)

    plt.title(r'Convergence Curve for $\alpha=$' + str(alpha) + ' and ' + str(numberIterations) + ' iterations')
    plt.ylabel('Cost Function Value')
    plt.xlabel('Number of Iterations')
    plt.show()

# Renders the circle figure with each of the given point marked as correct or not
def drawCirclePlot(x,predicted_results):
    fig, ax = plt.subplots()
    circ = plt.Circle((10, 10), radius=6, color='g', fill=False)
    ax.add_artist(circ)
    plt.title("Neural Network classification for test data")
    plt.xlim([0, 20])
    plt.ylim([0, 20])
    plt.xlabel("X")
    plt.ylabel("Y")
    for i in range(len(x)):
        if (predicted_results[i]) : plt.plot(x[i][0],x[i][1],'ro')
        else: plt.plot(x[i][0],x[i][1],'bv')
    red_patch = mpatches.Patch(color='red', label='Correct Classification',)
    blue_patch = mpatches.Patch(color='blue', label='Wrong Classification', )
    plt.legend(handles=[red_patch,blue_patch],bbox_to_anchor=(0.0, 0.98), ncol = 1,loc=2, borderaxespad=0.)

    plt.show()

if __name__ == '__main__':

    alpha = float(sys.argv[1])
    n_in,n_hidden,n_out = 2,int(sys.argv[2]),1
    num_iter = int(sys.argv[3])
    file_name = (sys.argv[4])

    # ------------------- Train ------------------
    train_set = np.loadtxt(file_name)
    x,y = network.extractData(train_set,n_out)
    x = network.normalize(x)
    nn = network.createNeuralNetwork(n_in,n_hidden,n_out)
    error,log = network.backPropagation(nn,x,y,alpha,num_iter)


    # -------------- TEST--------------------
    test_set = np.loadtxt("generated_10000.txt")
    x, y = network.extractData(test_set, n_out)
    x1 = network.normalize(x)
    good_predictions, bad_predictions, fp ,fn = 0, 0, 0, 0

    plot_pool_size = 100 # Number of point to be taken for the circle plot
    inside_counter,outside_counter = 0,0 # Keeps score of how many points have been stored for the plot
    circle_plot_point_coordinate, circle_plot_prediction = [], [] #Coordenates of the points and weather or not the prediction was correct

    print "\nID     Instance        Prediction Result  "
    for i in range(len(x1)):
        out = network.predict(nn, x1[i])
        result = round(out[0])
        correct = (result == y[i])

        # Store points for the plot
        if (inside_counter != plot_pool_size) and (y[i] == 1) :
            circle_plot_point_coordinate.append(x[i])
            circle_plot_prediction.append(correct)
            inside_counter +=1
        elif (outside_counter != plot_pool_size) and (y[i] == 0) :
            circle_plot_point_coordinate.append(x[i])
            circle_plot_prediction.append(correct)
            outside_counter +=1

        if (correct):
            good_predictions += 1
        else:
            bad_predictions += 1
            if (result == 1) : fp += 1
            else : fn += 1
        print str(i + 1) + " " + str(x[i]) + " " + str(result) + " " + str(correct)
    print "------------------ Results ---------------------"
    print ("File = %s Iterations = %d     Learning Rate = %.3f  Hidden Layer = %d") % (file_name,num_iter, alpha,n_hidden)
    print ("Error Value = %.4f") % (error)
    print ("Correctly Predicted = %d     Poorly Predicted = %d") % (good_predictions, bad_predictions)
    print ("False Positives = %d     False Negatives = %d") % (fp, fn)
    costFunctionPlot(log,num_iter,alpha)
    drawCirclePlot(circle_plot_point_coordinate, circle_plot_prediction)
