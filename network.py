#!/usr/bin/env python

'''
Small changes in implementation of network2 from Michael Nielsen's
neuralnetworksanddeeplearning online textbook code.
    - Add ability to run visual of training
    - Add compatibility with python 3
'''

import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import mnist_loader

__author__ = 'Jacob Manning'
__version__ = '0.1'

class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * 
            np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return (a - y)

class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weight_bias_initializer()
        self.cost = cost

    def weight_bias_initializer(self):
        # init a bias vector for each of the non-input layers 
        # with dim (y, 1) where y is the number of neurons
        # in that layer 
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # init a weight matrix for synapses between each layer 
        # with dim (y, x) where y is the number of neurons in the 
        # current layer and x is the number of neurons in the next layer
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in 
                        zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        # propogate the image through the network continuously
        # updating a as it goes through the network layers
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def sgd(self, train_data, epochs, mini_batch_size, learning_rate, 
            lmbda=0.0, eval_data=None, monitor_eval_cost=False, 
            monitor_eval_accuracy=False, monitor_train_cost=False,
            monitor_train_accuracy=False, interactive=False):

        # if training in interactive mode, create a figure
        if interactive:
            figure = plt.figure(figsize=(8, 4))
            plt.ion()
        else:
            figure = None

        if eval_data:
            n_data = len(eval_data)

        n = len(train_data)
        eval_cost, eval_accuracy = [], []
        train_cost, train_accuracy = [], []

        for j in range(epochs):
            # shuffle the training data
            np.random.shuffle(train_data)

            # split train data into mini batches
            mini_batches = [
                train_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for idx, mini_batch in enumerate(mini_batches):
                # display an example every 100 examples
                if figure and idx % 100 == 0:
                    display_example(self, mini_batch, figure)

                # calculate gradients
                self.update_mini_batch(mini_batch, learning_rate, 
                    lmbda, len(train_data))

            print('\nEpoch {} training complete'.format(j))

            if monitor_train_cost:
                cost = self.total_cost(train_data, lmbda)
                train_cost.append(cost)
                print('Cost on training data:', cost)

            if monitor_train_accuracy:
                accuracy = self.accuracy(train_data, convert=True)
                train_accuracy.append(accuracy)
                print('Accuracy on training data: {} / {}'.format(accuracy, n))

            if monitor_eval_cost:
                cost = self.total_cost(eval_data, lmbda, convert=True)
                eval_cost.append(cost)
                print('Cost on eval data:', cost)

            if monitor_eval_accuracy:
                accuracy = self.accuracy(eval_data)
                eval_accuracy.append(accuracy)
                print('Accuracy on eval data: {} / {}'.format(
                    accuracy, n_data))

        return eval_cost, eval_accuracy, train_cost, train_accuracy

    def update_mini_batch(self, mini_batch, learning_rate, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # calculate new gradients
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # update weights and biases based on gradients
        self.weights = [(1 - learning_rate * (lmbda / n)) * w - 
            (learning_rate / len(mini_batch)) * nw for w, nw in zip(
            self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb 
            for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            sp = sigmoid_prime(zs[-l])
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                for (x, y) in data]

        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0

        for x, y in data:
            a = self.feedforward(x)

            if convert:
                y = vectorized_result(y)

            cost += self.cost.fn(a, y) / len(data)

        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2
            for w in self.weights)

        return cost

    def save(self, filename):
        # save the network parameters in a dict
        data = {
            'sizes': self.sizes,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }

        # json dump the dict with all network params
        with open(filename, 'w') as f:
            json.dump(data, f)

    def test_net(self, test_data, iterations=100):
        figure = plt.figure(figsize=(8, 4))
        plt.ion()

        for _ in range(iterations):
            try:
                display_example(self, test_data, figure)
                time.sleep(5)
            except KeyboardInterrupt:
                plt.close('all')
                print()
                break

        plt.close()

def load(filename):
    # json load the values
    with open(filename, 'r') as f:
        data = json.load(f)

    # re-init the network
    net = Network(data['sizes'])
    # load the weights and biases
    net.weights = [np.array(w) for w in data['weights']]
    net.biases = [np.array(b) for b in data['biases']]

    return net

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    
    return e

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def view_example(net, data, figure, index=None):
    # if no index is specified, get a random one
    if not index:
        index = np.random.randint(0, len(data))

    # image: np array -> pixel values
    # label: int -> correct output
    image, label = data[index]
    if not isinstance(label, np.int64):
        label = np.argmax(label)

    # feed the image through the net to get the output layer
    prediction = net.feedforward(image)

    description1 = 'Label: {} '.format(label)
    description2 = 'Prediction: {}'.format(np.argmax(prediction))

    # add the two subplots to axes
    ax1 = figure.add_subplot(121)
    ax2 = figure.add_subplot(122)

    # reshape the image np array to the 28x28 size
    # grayscale the image
    ax1.matshow(np.reshape(image, (28, 28)), cmap='gray')
    ax1.set_title(description1)
    # remove the axes on the image plot
    ax1.axis('off')

    # the values for the x axis should be the ints from 0-9
    x_coords = np.arange(len(prediction))
    # plot a bar graph with x values 0-9 and y values of the prediction
    #   array that was from the feedforward of the nn
    ax2.bar(left=x_coords, height=prediction, tick_label=x_coords)
    ax2.set_title(description2)

def display_example(net, test_data, figure):
    # clear the display
    figure.clf()

    # helper function to establish axes
    view_example(net, test_data, figure)

    # draw the axes and show
    figure.canvas.draw()
    figure.show()

def main(filename):
    # load the data sets
    train_data, valid_data, test_data = mnist_loader.load_data_wrapper()

    # initialize the neural net
    #   the list parameter defines the # of neurons in each layer
    net = Network([784, 100, 10])
    # train the network using stochastic gradient descent
    net.sgd(train_data, 30, 10, 0.1, lmbda=10.0,
            eval_data=valid_data, monitor_eval_accuracy=True,
            monitor_eval_cost=True, monitor_train_accuracy=True,
            monitor_train_cost=True, interactive=True)
    # save the network in a json
    net.save(filename)

    # test the network on the test set
    net.test_net(test_data, iterations=20)

if __name__ == '__main__':
    filename = input('Network name\n> ')
    filename += '.json'
    main(filename)
    print()
    print('*' * 40)
    print('Success! Network saved in:', filename)
    print('*' * 40)
