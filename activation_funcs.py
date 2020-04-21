import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(z):
    # print(z)
    # print(np.max(z))
    z -= np.max(z)
    # print(z)
    sig = 1.0/(1.0+np.exp(-z))
    print(sig)
    return

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

def softmax(z):
    # z -= np.max(z)
    # sm = (np.exp(z).T / np.sum(np.exp(z), axis=0)).T
    # print(sm.shape)
    # return sm
    expZ = np.exp(z - np.max(z))
    sm = expZ / np.sum(expZ)
    # print(sm)
    # print(sm.shape)
    return sm

def softmax_prime(z):
    # initialize the 2-D jacobian matrix.
    # print(z.shape)
    # print('jacobian')
    # jacobian_m = np.diag(z)
    # for i in range(len(jacobian_m)):
    #     for j in range(len(jacobian_m)):
    #         if i == j:
    #             print(jacobian_m)
    #             print('a:' + str(z[i] * (1 - z[i])))
    #             jacobian_m[i][j] = z[i] * (1 - z[i])
    #         else:
    #             jacobian_m[i][j] = -z[i] * z[j]
    # return jacobian_m

    s = z.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


