from typing import List
from copy import deepcopy
import numpy as np

from RedWineClassifier.operations import *

class NeuralNetwork():

    def __init__(self, n_features: int, layer_sizes: List[int], activations: List[Activation], loss: Loss,
                 learning_rate: float=0.01, W_init: List[np.ndarray]=None):

        sizes = [n_features] + layer_sizes
        if W_init:
            assert all([W_init[i].shape == (sizes[i] + 1, sizes[i+1]) for i in range(len(layer_sizes))]), \
                "Specified sizes for layers do not match sizes of layers in W_init"
        assert len(activations) == len(layer_sizes), \
            "Number of sizes for layers provided does not equal the number of activations provided"

        self.n_layers = len(layer_sizes)
        self.activations = activations
        self.loss = loss
        self.learning_rate = learning_rate
        self.W = []
        for i in range(self.n_layers):
            if W_init:
                self.W.append(W_init[i])
            else:
                rand_weights = np.random.randn(sizes[i], sizes[i+1]) / np.sqrt(sizes[i])
                biases = np.zeros((1, sizes[i+1]))
                self.W.append(np.concatenate([biases, rand_weights], axis=0))

    def forward_pass(self, X) -> (List[np.ndarray], List[np.ndarray]):
        '''
        Executes the forward pass of the network on a dataset of n examples with f features. Inputs are fed into the
        first layer. Each layer computes Z_i = g(A_i) = g(Z_{i-1}W[i]).
        :param X: The training set, with size (n, f)
        :return A_vals: a list of a-values for each example in the dataset. There are n_layers items in the list and
                        each item is an array of size (n, layer_sizes[i])
                Z_vals: a list of z-values for each example in the dataset. There are n_layers items in the list and
                        each item is an array of size (n, layer_sizes[i])
        '''

        A_vals = []
        Z_vals = []

        for layer_index, weight_matrix in enumerate(self.W):
            layer = np.zeros((len(X), weight_matrix.shape[1]))
            for example_index, example in enumerate(X):
                example_layer = np.zeros(weight_matrix.shape[1])
                for node_index in range(weight_matrix.shape[1]): # nodes in current layer
                    sum = 0
                    for prev_node_index in range(weight_matrix.shape[0]): # nodes in previous layer, for the incoming edges
                        if layer_index == 0: # first layer whose weights get multiplied with input features
                            Xi = example[prev_node_index - 1] if prev_node_index > 0 else 1
                        else: # every other layer
                            Xi = Z_vals[layer_index - 1][example_index][prev_node_index - 1] if prev_node_index > 0 else 1

                        sum += (Xi * weight_matrix[prev_node_index][node_index])
                    example_layer[node_index] = sum
                layer[example_index] = example_layer
            A_vals.append(layer)
            Z_vals.append(self.activations[layer_index].value(layer))

        return A_vals, Z_vals

    def backward_pass(self, A_vals, dLdyhat) -> List[np.ndarray]:
        '''
        Executes the backward pass of the network on a dataset of n examples with f features. The delta values are
        computed from the end of the network to the front.
        :param A_vals: a list of a-values for each example in the dataset. There are n_layers items in the list and
                       each item is an array of size (n, layer_sizes[i])
        :param dLdyhat: The derivative of the loss with respect to the predictions (y_hat), with shape (n, layer_sizes[-1])
        :return deltas: A list of delta values for each layer. There are n_layers items in the list and
                        each item is an array of size (n, layer_sizes[i])
        '''

        deltas = []

        for layer_index in reversed(range(self.n_layers)):
            if layer_index < self.n_layers - 1: # every layer except last one
                next_weight_matrix = self.W[layer_index + 1]
                delta_layer = np.zeros_like(A_vals[layer_index])

                for example_index in range(A_vals[layer_index].shape[0]): # number of examples
                    sum_layer = np.zeros(next_weight_matrix.shape[1])
                    for next_node_index in range(next_weight_matrix.shape[1]): # nodes in next layer
                        for node_index in range(next_weight_matrix.shape[0]): # nodes in current layer
                            sum_layer[next_node_index] += deltas[0][example_index][next_node_index] * next_weight_matrix[node_index][next_node_index]
                        
                        delta_layer[example_index] = sum_layer[next_node_index] * self.activations[layer_index].derivative(A_vals[layer_index])[example_index]
            else: # last layer
                delta_layer = dLdyhat * self.activations[layer_index].derivative(A_vals[layer_index])

            deltas.insert(0, delta_layer)

        return deltas

    def update_weights(self, X, Z_vals, deltas) -> List[np.ndarray]:
        '''
        Having computed the delta values from the backward pass, update each weight with the sum over the training
        examples of the gradient of the loss with respect to the weight.
        :param X: The training set, with size (n, f)
        :param Z_vals: a list of z-values for each example in the dataset. There are n_layers items in the list and
                       each item is an array of size (n, layer_sizes[i])
        :param deltas: A list of delta values for each layer. There are n_layers items in the list and
                       each item is an array of size (n, layer_sizes[i])
        :return W: The newly updated weights (i.e. self.W)
        '''

        temp_W = deepcopy(self.W)

        for layer_index, delta_layer in enumerate(deltas): # iterate through layers
            for node_index in range(delta_layer.shape[1]): # iterate through each node
                for prev_node_index in range(temp_W[layer_index].shape[0]): # iterate through each incoming edge of a node in current layer
                    error_deriv = 0
                    for example_index in range(X.shape[0]): # iterate through each example
                        if layer_index == 0: # input layer
                            temp = X[example_index][prev_node_index - 1] if prev_node_index > 0 else 1
                            error_deriv += (delta_layer[example_index][node_index] * temp)
                        else:
                            temp = Z_vals[layer_index - 1][example_index][prev_node_index - 1] if prev_node_index > 0 else 1
                            error_deriv += (delta_layer[example_index][node_index] * temp)

                    temp_W[layer_index][prev_node_index][node_index] -= (self.learning_rate * error_deriv) # update the weight
                
        return temp_W

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int) -> (List[np.ndarray], List[float]):
        '''
        Trains the neural network model on a labelled dataset.
        :param X: The training set, with size (n, f)
        :param y: The targets for each example, with size (n, 1)
        :param epochs: The number of epochs to train the model
        :return W: The trained weights
                epoch_losses: A list of the training losses in each epoch
        '''

        epoch_losses = []
        for epoch in range(epochs):
            A_vals, Z_vals = self.forward_pass(X)   # Execute forward pass
            y_hat = Z_vals[-1]                      # Get predictions
            L = self.loss.value(y_hat, y)           # Compute the loss
            print("Epoch {}/{}: Loss={}".format(epoch, epochs, L))
            epoch_losses.append(L)                  # Keep track of the loss for each epoch

            dLdyhat = self.loss.derivative(y_hat, y)         # Calculate derivative of the loss with respect to output
            deltas = self.backward_pass(A_vals, dLdyhat)     # Execute the backward pass to compute the deltas
            self.W = self.update_weights(X, Z_vals, deltas)  # Calculate the gradients and update the weights

        return self.W, epoch_losses

    def evaluate(self, X: np.ndarray, y: np.ndarray, metric) -> float:
        '''
        Evaluates the model on a labelled dataset
        :param X: The examples to evaluate, with size (n, f)
        :param y: The targets for each example, with size (n, 1)
        :param metric: A function corresponding to the performance metric of choice (e.g. accuracy)
        :return: The value of the performance metric on this dataset
        '''

        A_vals, Z_vals = self.forward_pass(X)       # Make predictions for these examples
        y_hat = Z_vals[-1]
        metric_value = metric(y_hat, y)     # Compute the value of the performance metric for the predictions
        return metric_value

