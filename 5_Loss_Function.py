# Log loss --
import math
import numpy as np

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = - (math.log(softmax_output[0])*target_output[0] +
            math.log(softmax_output[1])*target_output[1] +
            math.log(softmax_output[2])*target_output[2])
print(loss)

# But log loss doesn't account for the model's confidence in its prediction, where cross entropy does
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                    [0.1, 0.5, 0.4],
                    [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]

# for targ_idx, distribution in zip(class_targets, softmax_outputs):
#     print(distribution[targ_idx])

# print(softmax_outputs[[0, 1, 2], class_targets])

# Print the probabilities of the predicted class and then calculate the log loss
print(softmax_outputs[range(len(softmax_outputs)), class_targets])
print(-np.log(softmax_outputs[range(len(softmax_outputs)), class_targets]))

# We have the losses for each class, but we want a mean loss for the model
neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
average_loss = np.mean(neg_log)
print(average_loss)

# But we can one-hot encode our class targets, so we'll need a conditional to handle those cases
import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 1, 0]])

# Probabilities for target values - if categorical labels
if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[range(len(softmax_outputs)), class_targets]
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(softmax_outputs * class_targets, axis=1)

# Losses
neg_log = -np.log(correct_confidences)
average_loss = np.mean(neg_log)
print(average_loss)

# Accuracy Calculation
import numpy as np

softmax_outputs = np.array([[0.7, 0.2, 0.1],
                            [0.5, 0.1, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([0, 1, 1])

# Calculate values along second axis
predictions = np.argmax(softmax_outputs, axis=1)
if len(class_targets.shape) == 2:
    class_targets = np.argmax(class_targets, axis=1)
accuracy = np.mean(predictions == class_targets)
print('Accuracy: ', accuracy)