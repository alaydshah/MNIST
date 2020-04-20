from layer import Layer

# inherit from base class Layer

class ActivationLayer(Layer):

    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propogation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY
    # learning rate is not used because there are no learnable parameters
    def backward_propogation(self, output_error):
        return self.activation_prime(self.input) * output_error

    def update_parameters(self, learning_rate, batch_size):
        pass

