# coding=utf-8
import numpy as np
import NN as network
import sys

# Universidad Simón Bolívar
# Inteligencia Artificial II
# Prof. Ivette Carolina Martínez
# Alumnos:
# Carlos Martinez 11-10584
# Yerson Roa 11-10876
# Antonio Scaramazza 11-10957

# splits the data into two separate sets
def splitData(data,percentage,output_num):
    cut_index = int(round((percentage/float(100))*data.shape[0]))
    data_range = int(round(cut_index/float(output_num)))
    tmp = data.shape[0]/output_num
    data_train,data_test = [],[]
    for i in range(output_num):
        tmp2 = data[i*(tmp):data_range+i*tmp]
        for j in tmp2:
            data_train.append((j.tolist()))
        tmp3 = data[data_range+i*tmp:(i+1)*tmp]
        for j in tmp3: data_test.append((j.tolist()))
    data_train = np.resize(data_train,(len(data_train),len(data_train[0])))
    data_test = np.resize(data_test,(len(data_test),len(data_train[0])))
    return data_train,data_test

# Converts the result from the neural network prediction into the format of the classifier,
# that is a vector whose value is either 1 or 0. Eg: [1,0]
def convert (result):
    index = np.argmax(result)
    out = []
    for i in range(len(result)):
        if (i == index) : out += [1]
        else: out+=[0]
    return out

if __name__ == '__main__':

    alpha = float(sys.argv[1])
    n_in, n_hidden, n_out = 4, int(sys.argv[2]), int(sys.argv[3])
    num_iter = int(sys.argv[4])
    file_name = sys.argv[5]
    percentage = int(sys.argv[6]) # Data split percentage
    data = np.loadtxt(file_name, delimiter=",")
    train_data, test_data = splitData(data, percentage, n_out)


    # -----------------TRAIN-------------------
    x,y = network.extractData(train_data, n_out)
    nn = network.createNeuralNetwork(n_in,n_hidden,n_out)
    error,log = network.backPropagation(nn,x,y,alpha,num_iter)

    # -------------- TEST--------------------
    x, y = network.extractData(test_data, n_out)
    good_predictions,bad_predictions = 0,0
    print "\nID     Instance        Prediction Result  "
    for i in range(len(x)):
        out = network.predict(nn, x[i])
        result = convert(out)
        correct = np.array_equal(result,y[i])
        if (correct) : good_predictions += 1
        else:  bad_predictions += 1
        print str(i+1) + " " + str(x[i]) + " " + str(result) + " " + str(correct)
    print "------------------ Results ---------------------"
    print ("Iterations = %d     Learning Rate = %.3f") % (num_iter, alpha)
    print ("Error Value = %.4f"  ) % (error)
    print ("Correctly Predicted = %d     Badly Predicted = %d" ) % (good_predictions,bad_predictions)