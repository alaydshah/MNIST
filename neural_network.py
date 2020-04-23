import random
import numpy as np

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propogation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propogation(output)
            result.append(output)

        return result

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.predict(x)), y) for (x,y) in test_data]
        # print(test_results)
        return sum(int(x == y) for (x,y) in test_results)

    # train the network
    def fit(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        # sample dimension first
        if test_data:
            n_test = len(test_data)

        n_train = len(training_data)
        n_test = len(test_data)

        # training loop
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n_train, mini_batch_size)]
            err = 0
            for mini_batch in mini_batches:
                batch_err = 0
                for j in range(mini_batch_size):

                    # forward propogation
                    train_example = mini_batch[j]
                    output = train_example[0]
                    label = train_example[1]

                    # output = output[None, :]
                    for layer in self.layers:
                        output = layer.forward_propogation(output)

                    # compute loss (for display)
                    # new_err = self.loss(label, output)
                    batch_err += self.loss(label, output)
                    # print(new_err)

                    # backward propogation
                    error = self.loss_prime(label, output)
                    for layer in reversed(self.layers):
                        error = layer.backward_propogation(error)

                for layer in self.layers:
                    layer.update_parameters(learning_rate, mini_batch_size)
                err += batch_err
                print(batch_err/mini_batch_size)
            # calculate average error on all samples
            err /= n_train
            evaluation = self.evaluate(test_data)
            print('epoch %d/%d error=%f, evaluation=%d/%d' % (i+1, epochs, err, evaluation, n_test))