#!/usr/bin/env python

'''
Small changes in implementation of mnist_loader from Michael Nielsen's
neuralnetworksanddeeplearning online textbook code.
    - Add compatibility with python 3
'''

import pickle
import gzip
import numpy as np

def load_data():
    with gzip.open('data/mnist.pkl.gz', mode='rb') as f:
        training_data, validation_data, test_data = pickle.load(
            f, encoding='latin1')

    return (training_data, validation_data, test_data)

def load_data_wrapper():
    # call load_data to get tuple of data sets
    tr_d, va_d, te_d = load_data()

    # training inputs are the first column of the training_data in a 784x1 vec
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    # the result should be a col vector with a 1 at the correct output neuron 
    # and zeros everywhere else
    training_results = [vectorized_result(y) for y in tr_d[1]]
    # training_inputs and training_results become one vector with inputs in one
    # col and results in the other
    training_data = list(zip(training_inputs, training_results))

    # valid. inputs are the first column of the validation_data in a 784x1 vec
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    # va_inputs and va_d become one vector as they did in training_results
    validation_data = list(zip(validation_inputs, va_d[1]))

    # same as validation
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    # tuple of the three data sets
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    
    return e