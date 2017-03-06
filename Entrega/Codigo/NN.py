# coding=utf-8
import numpy as np
import random
import math

# Universidad Simón Bolívar
# Inteligencia Artificial II
# Prof. Ivette Carolina Martínez
# Alumnos:
# Carlos Martinez 11-10584
# Yerson Roa 11-10876
# Antonio Scaramazza 11-10957

# Back propagation algorithm extracted from Tom Mitchell's Machine Learning

# A neural network is represented by a list of layer
# A layer is a list of neurons
# A neuron contains the following values:
#        First N values (where N is the number of inputs from previous layer) are the weights chosen at random at
#        first
#        The N+1 value is the bias also chosen at random
#        The last 2 values are the output of the neuron and the error respectably

# Returns a layer with weights and bias chosen at random, plus 0 value error and output by default
def createNeuronLayer(n, size):
    layer = []
    for i in range(0, n):
        neuron = []
        for j in range(size+1):
            neuron.append(random.uniform(-0.5, 0.5))
        neuron.append(0) # Add zero output value as default
        neuron.append(0) # Add zero error value as default
        layer.append(neuron)
    return layer

# Returns a neural network with n_hidden neurons in its hidden layer
# and n_out network in its outputs layer
def createNeuralNetwork(n_in, n_hidden, n_out):
    network = [createNeuronLayer(n_hidden, n_in), createNeuronLayer(n_out, n_hidden)]
    return network

# Returns the cost for the weights and inputs
def calculateCost(weights,input):
    return np.dot(weights,input)

# Sigmoid function value to fire the neuron
def activateSigmoid(cost):
    return 1.0 / (1.0 + math.exp(-cost))

# Calculate the output of every neuron, store it, and return the final output of the network given a learning instance
def forwardPropagate(nn,instance):
    input = instance
    for layer in nn :
        new_values = []
        for neuron in layer:
            net = calculateCost(neuron[:-3],input) + neuron[-3]
            sigmoid_result = activateSigmoid(net)
            neuron[-2] = sigmoid_result #Store output in the neuron
            new_values.append(sigmoid_result) # Keep a vector with the outputs of the layer
        input = new_values
    return input

# Calculate the prediction for a test instance
def predict(nn, instance):
    input = instance
    for layer in nn :
        new_values = []
        for neuron in layer:
            net = calculateCost(neuron[:-3],input) + neuron[-3]
            sigmoid_result = activateSigmoid(net)
            new_values.append(sigmoid_result)
        input = new_values
    return input

# Calculate the errors in the output layer
def calculateErrorOutputs(nn,expected):
    i = 0
    layer = nn[-1]
    for neuron in layer  :
        expected_value = expected[i]
        output = neuron[-2]
        neuron[-1] = output*(1-output)*(expected_value-output)
        i += 1

# Calculate the errors in the hidden layer
def calculateErrorHidden(nn):
    hidden_layer = nn[0]
    output_layer = nn[1]
    for neuron in hidden_layer :
        j = hidden_layer.index(neuron)
        output_hidden = neuron[-2]
        total = 0
        for i in range(len(output_layer)):
            total += output_layer[i][-1] * output_layer[i][j]
        neuron[-1] = output_hidden*(1-output_hidden)*total

# Update the weights of every network using the error calculated previously
def updateWeights(nn,alpha,instance):
    for i in range(0,len(nn)):
        layer = nn[i]
        for j in range(0,len(layer)):
            neuron = layer[j]
            for k in range(len(neuron)-3):
                if i == 0:
                    delta = alpha*neuron[-1]*instance[k]
                    neuron[k] = neuron[k] + delta
                else:
                    delta = alpha * neuron[-1] * nn[i-1][k][-2]
                    neuron[k] = neuron[k] + delta

# Train the neural network with the training set compose of vector x and y
# Returns the final SSE during training
def backPropagation(nn, x,y, alpha, num_iter):
    log = []
    n = len(x)
    for i in range(0, num_iter):
        sum_error,cost_func = 0,0
        for k in range(len(x)):
            instance = x[k]
            expected = y[k]
            outputs = forwardPropagate(nn, instance)
            sum_error += (sum([(expected[j] - outputs[j]) ** 2 for j in range(len(outputs))]))
            cost_func += sum_error/(2*n)
            calculateErrorOutputs(nn, expected)
            calculateErrorHidden(nn)
            updateWeights(nn, alpha, instance)
        if (i % 10) == 0 :
            print('Iter=%d, alpha=%.3f, error=%.3f' % (i, alpha, sum_error))
            log.append([i,cost_func])
    log = np.resize(log,(len(log),len(log[0])))
    return sum_error,log

# Split the data set into two vectors, 'x' contains the instances
# and 'y' contains the labels for each one of them
def extractData(data,y_size):
    numberColumns = data.shape[1]
    y = data[:,numberColumns-y_size:]
    x = data[:, 0:numberColumns - y_size]
    return x,np.reshape(y,(len(y),y_size))

# Normalize the data given using the statistical form
def normalize(array):
    n = len(array)
    media = np.mean(array,axis=0)
    dev_standard = np.std(array,axis=0)

    #Create normalized array
    new_array = []
    for elem in array:
        temp = np.divide((elem - media) ,dev_standard)
        new_array.append(temp)
    out = np.asarray(new_array)
    return out


