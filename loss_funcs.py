import numpy as np

# loss function and its derivative
def mse(y_enc, output):
    return np.mean(np.power(y_enc-output, 2))

def mse_prime(y_enc, output):
    return 2*(output-y_enc)/y_enc.size;

def log_likelihood(y_enc, output):
    return -np.log(np.sum(y_enc * output))

def log_likelihood_prime(y_enc, output):
    return -y_enc/np.sum(y_enc * output)



