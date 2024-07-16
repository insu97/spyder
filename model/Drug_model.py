# -*- coding: utf-8 -*-
#%% import library
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from function.DeepLearning import relu, softmax

#%% 01. drug_model
def init_network():
    network = {}
    network['W1'] = 0.01 * np.random.randn(9, 6)
    network['b1'] = np.zeros(6)
    network['W2'] = 0.01 * np.random.randn(6, 6)
    network['b2'] = np.zeros(6)
    network['W3'] = 0.01 * np.random.randn(6, 5)
    network['b3'] = np.zeros(5)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = relu(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = relu(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y
     