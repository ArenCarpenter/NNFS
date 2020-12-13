# Vid 4: Batches, Layers, Objects
import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# np.dot won't work here because our shapes of inputs and weights don't match! Need to transpose
# output = np.dot(weights, inputs) + biases

# need to transpose -- original shapes of (3,4) (3,4) doesn't work because [1] of one needs to match [0] of the other
# and 3 != 4, after transposition, (3,4) (4,3) now does have 4 == 4

output = np.dot(inputs, np.array(weights).T) + biases
print(output)

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)
