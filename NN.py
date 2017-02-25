import numpy as np
import random
import math


def createNeuronLayer(n, size):
    layer = []
    for i in range(0, n):
        neuron = []
        for j in range(0, size):
            neuron.append(random.uniform(-0.5, 0.5))
        neuron.append(0)
        neuron.append(0)
        layer.append(neuron)
    return layer


def createNeuralNetwork(n_in, n_hidden, n_out):
    network = [createNeuronLayer(n_hidden, n_in), createNeuronLayer(n_out, n_hidden)]
    return network

def calculateCost(weights,input):
    return np.dot(weights,input)

def activateSigmoid(cost):
    return 1.0 / (1.0 + math.exp(-cost))

def forwardPropagate(nn,instance):
    input = instance
    for layer in nn :
        new_values = []
        for neuron in layer:
            net = calculateCost(neuron[:-2],input)
            sigmoid_result = activateSigmoid(net)
            neuron[-2] = sigmoid_result
            new_values.append(sigmoid_result)
        input = new_values
    return input

def calculateErrorOutputs(nn,expected):
    i = 0
    layer = nn[-1]
    for neuron in layer  :
        expected_value = expected[i]
        output = neuron[-2]
        neuron[-1] = output*(1-output)*(expected_value-output)
        i += 1

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

def updateWeights(nn,alpha,instance):
    for i in range(0,len(nn)):
        layer = nn[i]
        for j in range(0,len(layer)):
            neuron = layer[j]
            for k in range(len(neuron)-2):
                if i == 0:
                    delta = alpha*neuron[-1]*instance[k]
                    neuron[k] = neuron[k] + delta
                else:
                    delta = alpha * neuron[-1] * nn[i-1][j][-2]
                    neuron[k] = neuron[k] + delta


def backPropagation(nn,train_data,alpha,numIter):

    for i in range(0,numIter):
        for row in train_data:
            instance = row[:-1]
            expected = row[-1]
            forwardPropagate(nn, instance)
            calculateErrorOutputs(nn, [expected])
            calculateErrorHidden(nn)
            updateWeights(nn, alpha, instance)


if __name__ == '__main__':

    nn = createNeuralNetwork(2, 3, 1)
    # nn = [[[0.8, 0.2, 0, 0],
    #        [0.4, 0.9, 0, 0],
    #        [0.3, 0.5, 0, 0]],
    #       [[0.3, 0.5, 0.9, 0, 0]]]

    print nn
    data = [[1,1,1],
            [1,0,1],
            [0,1,1],
            [0,0,0]]
    backPropagation(nn,data,0.8,100000)
    print nn
    print forwardPropagate(nn,[0,0])