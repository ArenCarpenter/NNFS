# making an object
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
nnfs.init()

# Function to create spiral data set of 100 pts in 3 classes
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

# spiral_data is imported from NNFS package
X, Y = spiral_data(100, 3)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)  # 0.10 to scale, define as inp, neur so no transpose
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()
layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)

# Activation functions
# Step function 1 if positive, 0 if negative
# Sigmoid function, more reliable bec granularity, vanishing gradient problem
# ReLu, x>0 output is x, if x<0 output is 0, fast, can fit non-linear relationships

# inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
# output = []

# One option for ReLU:
# for i in inputs:
#     if i > 0:
#         output.append(i)
#     elif i <= 0:
#         output.append(0)

# A better option with np.max()!
# output.append(max(0, i)) this also works
