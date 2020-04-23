import numpy as np

# loss function and its derivative
def mse(y_enc, output):
    # print(y_enc)
    # print(output)
    return np.mean(np.power(y_enc-output, 2))

def mse_prime(y_enc, output):
    return 2*(output-y_enc)/y_enc.size;

def log_likelihood(y_enc, output):
    # print(y_enc)
    # print(output)
    # return -np.log(np.dot(y_enc.flatten(), output.flatten()))
    # cost =  -np.mean(y_enc * np.log(output + 1e-8))
    # return cost
    # print(output.shape)
    # print(output + 1e-8)
    # print(np.log(output + 1e-8))

    cost = -np.mean(y_enc * np.log(output + 1e-8))
    return cost

def log_likelihood_prime(y_enc, output):
    return output - y_enc



