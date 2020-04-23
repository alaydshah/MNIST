from layer import Layer
import numpy as np


# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.delta_weights = np.zeros(shape=self.weights.shape)
        self.delta_bias = np.zeros(shape=self.bias.shape)

    # returns output for a given input
    def forward_propogation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW. dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propogation(self, output_error):
        # print("Output_Error: " + str(output_error))
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # print("Weights_error: " + str(weights_error))

        # update parameters
        self.delta_weights += weights_error
        self.delta_bias += output_error
        return input_error

    def update_parameters(self, learning_rate, batch_size):
        # print("Current Weights' Shapes:" + str(self.weights.shape))
        # print("Current Weights: " + str(self.weights))
        self.weights -= (learning_rate / batch_size) * self.delta_weights
        # print("Updated Weights: " + str(self.weights))
        # print("Current Bias: " + str(self.bias))
        self.bias -= (learning_rate / batch_size) * self.bias
        # print("Updated Bias: " + str(self.bias))
        self.delta_weights = np.zeros(shape=self.weights.shape)
        self.delta_bias = np.zeros(shape=self.bias.shape)
