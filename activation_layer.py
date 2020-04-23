from layer import Layer
import numpy as np

# inherit from base class Layer

class ActivationLayer(Layer):

    def __init__(self, activation, activation_prime, name):
        self.activation = activation
        self.activation_prime = activation_prime
        self.activation_name = name
    # returns the activated input
    def forward_propogation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY
    # learning rate is not used because there are no learnable parameters
    def backward_propogation(self, output_error):
        # print("Output_Error: " + str(output_error))
        if self.activation_name == 'softmax':
            # gradient = self.activation_prime(self.input)
            # back_prop_error = np.dot(output_error.reshape(1,10), gradient)
            back_prop_error = output_error.reshape(1,10)
        else:
            back_prop_error = self.activation_prime(self.input) * output_error
        # print(back_prop_error.shape)
        return back_prop_error

    def update_parameters(self, learning_rate, batch_size):
        pass
        #
