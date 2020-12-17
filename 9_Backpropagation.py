x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

z = xw0 + xw1 + xw2 + b

y = max(z, 0)

dvalue = 1.0

drelu_dz = dvalue * (1. if z > 0 else 0.)
print(drelu_dz)

dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1
drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db
print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]
drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2
print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

dx = [drelu_dx0, drelu_dx1, drelu_dx2]
dw = [drelu_dw0, drelu_dw1, drelu_dw2]
db = drelu_db

print(w, b)

w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * db
print(w, b)

xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

z = xw0 + xw1 + xw2 + b
y = max(z, 0)
print(y)

# Now for a layer of neurons
import numpy as np

# dvalues for one sample
# dvalues = np.array([[1., 1., 1.]])

# But we'll work in batches, so need multiple dvalue arrays
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

biases = np.array([[2, 3, 0.5]])

inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])

# dx0 = sum([weights[0][0] * dvalues[0][0], weights[0][1] * dvalues[0][1], weights[0][2] * dvalues[0][2]])
# dx1 = sum([weights[1][0] * dvalues[0][0], weights[1][1] * dvalues[0][1], weights[1][2] * dvalues[0][2]])
# dx2 = sum([weights[2][0] * dvalues[0][0], weights[2][1] * dvalues[0][1], weights[2][2] * dvalues[0][2]])
# dx3 = sum([weights[3][0] * dvalues[0][0], weights[3][1] * dvalues[0][1], weights[3][2] * dvalues[0][2]])

# Numpy can handle the element wise multiplication because both are np.arrays already
# dx0 = sum(weights[0] * dvalues[0])
# dx1 = sum(weights[1] * dvalues[0])
# dx2 = sum(weights[2] * dvalues[0])
# dx3 = sum(weights[3] * dvalues[0])

# And this can be simplified again with np.dot!
dinputs = np.dot(dvalues, weights.T)
dweights = np.dot(inputs.T, dvalues)
dbiases = np.sum(dvalues, axis=0, keepdims=True)
print(dinputs)
print(dweights)
print(dbiases)

# ReLU
z = np.array([[1, 2, -3, -4],
              [2, -7, -1, 3],
              [-1, 2, 5, -1]])
dvalues = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])

# drelu = np.zeros_like(z)
# drelu[z > 0] = 1 # if z is greater than 0, set to 1

# drelu *= dvalues # multiple by our boolean mask, neg become 0, and pos become their original value
drelu = dvalues.copy()
drelu[z <= 0] = 0
print(drelu) # see that all neg values are now 0

# Forward pass
layer_outputs = np.dot(inputs, weights) + biases
relu_outputs = np.maximum(0, layer_outputs)

drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0

dinputs = np.dot(drelu, weights.T)
dweights = np.dot(inputs.T, drelu)
dbiases = np.sum(drelu, axis=0, keepdims=True)

weights += -0.001 * dweights
biases += -0.001 * dbiases

print(weights)
print(biases)
