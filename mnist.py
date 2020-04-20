import numpy as np
import pickle
from neural_network import Network
from fully_connected_layer import FCLayer
from activation_layer import ActivationLayer
from activation_funcs import tanh, tanh_prime, sigmoid, sigmoid_prime
from loss_funcs import mse, mse_prime

# import gzip
# import os
# os.system("clear")
# import keras



# Function for one_hot encoding of output labels
# def one_hot_encoding(labels, num_labels=10):
#     one_hot = np.zeros((labels.shape[0], num_labels))
#     for row, col in enumerate(labels):
#         one_hot[row, col] = 1.0
#     return one_hot

def one_hot_encoding(label, num_labels=10):
    one_hot_vec = np.zeros(num_labels)
    one_hot_vec[label] = 1.0
    return one_hot_vec

def load_data(requested_data):

    dataset = ['train_image', 'train_label', 'test_image', 'test_label']

    if requested_data not in dataset:
        raise ValueError('Invalid Data Requested')

    data_file = requested_data + '.csv'
    pickle_file = requested_data + '.pkl'

    # print(data_file)
    # print(pickle_file)

    try:
        result_data = pickle.load(open(pickle_file, "rb"))
    except (OSError, IOError) as e:
        result_data = np.genfromtxt(data_file, delimiter=',', dtype=np.uint8)
        pickle.dump(result_data, open(pickle_file, "wb"))

    return result_data



# load data
training_images = load_data('train_image')
training_labels = load_data('train_label')
testing_images = load_data('test_image')
testing_labels = load_data('test_label')

training_images = training_images.astype('float32')
testing_images = testing_images.astype('float32')

# Verify dimensions of datasets
print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

# Normalizing data and One-hot encoding labels
training_images /= 255
# training_labels = one_hot_encoding(training_labels)
testing_images /= 255
# testing_labels = one_hot_encoding(testing_labels)

training_inputs = [np.reshape(x, (1, 28*28)) for x in training_images]
training_results = [one_hot_encoding(y) for y in training_labels]
training_data = list(zip(training_inputs, training_results))

testing_inputs = [np.reshape(x, (1, 28*28)) for x in testing_images]
testing_results = testing_labels.tolist()
testing_data = list(zip(testing_inputs, testing_results))


# print(len(training_data))
# print(len(testing_data))

# Neural Network
nn_net = Network()
nn_net.add(FCLayer(28*28, 100))                 # input_shape = (1, 28x28)  ;   output_shape = (1, 100)
nn_net.add(ActivationLayer(sigmoid, sigmoid_prime))
nn_net.add(FCLayer(100, 50))                    # input_shape = (1, 100)    ;   output_shape = (1, 50)
nn_net.add(ActivationLayer(sigmoid, sigmoid_prime))
nn_net.add(FCLayer(50, 10))                     # input_shape = (1, 50)     ;   output_shape = (1, 10)
nn_net.add(ActivationLayer(sigmoid, sigmoid_prime))

# train on training samples
nn_net.use(mse, mse_prime)
nn_net.fit(training_data, epochs=30, mini_batch_size=10, learning_rate=5.0, test_data=testing_data)

# # test on 3 samples
# out = nn_net.predict(testing_images[0:3])
# print("\n")
# print("predicted values : ")
# print(out, end="\n")
# print("true values : ")
# print(testing_labels[0:3])



# # print(x_train[1])
# print(x_test[1])
# # print(y_train[1])
# print(y_test[1])

# # training data 60000 samples
# # reshape and normalize input data

# x_train /= 255
#
# # encode output which is a number in range [0,9] into a vector of size 10
# y_train = one_hot_encoding(y_train)
# # y_train = np_utils.to_categorical(y_train)
# print(y_train.shape)

# print(pickle.load(open("names.dat", "rb")))

# Network
# neural_net = Network()
# neural_net.add(FCLayer())



# def load_data(requested_data):
#     train_data = np.genfromtxt('train_image.csv', delimiter=',', dtype=np.uint8)
#     train_labels = np.genfromtxt('train_label.csv', delimiter=',', dtype=np.uint8)
#     test_data = np.genfromtxt('test_image.csv', delimiter=',', dtype=np.uint8)
#     test_labels = np.genfromtxt('test_label.csv', delimiter=',', dtype=np.uint8)
#
#     print(train_data.shape)
#     print(train_labels.shape)
#     print(test_data.shape)
#     print(test_labels.shape)
#
#     return (train_data, train_labels), (test_data, test_labels)

