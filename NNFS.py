
# Using loops instead of multiple raw lists
# inputs = [1, 2, 3, 2.5]
#
# weights = [[0.2, 0.8, -0.5, 1],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]
#
# biases = [2, 3, 0.5]
#
# layer_outputs = []
# for neuron_weights, neuron_bias in zip(weights, biases):
#     neuron_output = 0
#     for n_input, weight in zip(inputs, neuron_weights):
#         neuron_output += n_input*weight
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)
#
# print(layer_outputs)

# Using Numpy!



# inputs = [1, 2, 3, 2.5]
# weights = [0.2, 0.8, -0.5, 1.0]
# bias = 2
#
# output = np.dot(inputs, weights) + bias
# print(output)

# inputs = [1, 2, 3, 2.5]
#
# weights = [[0.2, 0.8, -0.5, 1],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]
#
# biases = [2, 3, 0.5]
#
# output = np.dot(weights, inputs) + biases
# """first element is how return is indexed, we have 3 neurons so we want
# three outputs from the 3 sets of weights"""
# print(output)

# Vid 4: Batches, Layers, Objects
# inputs = [[1, 2, 3, 2.5],
#           [2.0, 5.0, -1.0, 2.0],
#           [-1.5, 2.7, 3.3, -0.8]]
#
# weights = [[0.2, 0.8, -0.5, 1],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]
#
# biases = [2, 3, 0.5]

# np.dot won't work here because our shapes of inputs and weights don't match! Need to transpose
# output = np.dot(weights, inputs) + biases

# need to transpose -- original shapes of (3,4) (3,4) doesn't work because [1] of one needs to match [0] of the other
# and 3 != 4, after tranposition, (3,4) (4,3) now does have 4 == 4

# output = np.dot(inputs, np.array(weights).T) + biases
# print(output)

# weights2 = [[0.1, -0.14, 0.5],
#             [-0.5, 0.12, -0.33],
#             [-0.44, 0.73, -0.13]]
#
# biases2 = [-1, 2, -0.5]
#
# layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
# layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
#
# print(layer2_outputs)

# making an object
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
nnfs.init()

# def create_data(points, classes):
#     X = np.zeros((points*classes, 2))
#     Y = np.zeros(points*classes, dtype='uint8')
#     for class_number in range(classes):
#         ix = range(points*class_number, points*(class_number+1))
#         r = np.linspace(0.0, 1, points) # radius
#         t = np.linspace(class_number*4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
#         X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
#         Y[ix] = class_number
#     return X, Y

# X, Y = create_data(100, 3)
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

X, Y = spiral_data(100, 3)

# X = [[1, 2, 3, 2.5],
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]]

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)  # 0.10 to scale, define as inp, neur so no transpose
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(2, 5)  # output here is input to layer2 so need to match
# layer2 = Layer_Dense(5, 2)
activation1 = Activation_ReLU()
layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)
# layer2.forward(layer1.output)


# Activation functions
# Step function 1 if positive, 0 if negative
# Sigmoid function, more reliable bec granularity, vanishing gradient problem
# ReLu, x>0 output is x, if x<0 output is 0, fast, can fit non-linear relationships

# inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
# output = []
#
# for i in inputs:
#     if i > 0:
#         output.append(i)
#     elif i <= 0:
#         output.append(0)

    # output.append(max(0, i)) this also works


