import numpy as np


def linear_regression_prediction(X, w):
    """
    Calculates the linear regression prediction.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :return: prediction
    :rtype: np.array(shape=(N, 1))
    """

    return X.dot(w)


def standardize(X):
    """
    Returns standardized version of the ndarray 'X'.

    :param X: input array
    :type X: np.ndarray(shape=(N, d))
    :return: standardized array
    :rtype: np.ndarray(shape=(N, d))
    """

    N = X.shape[0]
    d = X.shape[1]
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_out = np.zeros((N,d))
    for i in range(0, d):
        X_out[:,i] = (X[:,i] - mean[i])/std[i]

    return X_out


def compute_cost(X, y, w):
    """
    Calculates  mean square error cost.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d,))
    :return: cost
    :rtype: float
    """

    N = X.shape[0]
    diff = np.dot(X,w) - y # y_hat - y
    J = 1/N * np.dot(np.transpose(diff), diff)

    return J


def compute_wgrad(X, y, w):
    """
    Calculates gradient of J(w) with respect to w.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d,))
    :return: gradient
    :rtype: np.array(shape=(d,))
    """

    N = X.shape[0]
    diff = np.dot(X,w) - y
    grad = 2/N * np.dot(np.transpose(X), diff)

    return grad


def batch_gradient_descent(X, y, w, learning_rate, num_iters):
    """
     Performs batch gradient descent optimization.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d,))
    :param learning_rate: learning rate
    :type learning_rate: float
    :param num_iters: number of iterations
    :type num_iters: int
    :return: weights, weights history, cost history
    :rtype: np.array(shape=(d,)), list, list
    """

    weights_history = [w.flatten()]
    cost_history = [compute_cost(X, y, w)]

    w_current = w
    for t in range(0, num_iters-1):
        grad = compute_wgrad(X, y, w_current)
        w_current = w_current - learning_rate * grad
        weights_history.append(w_current.flatten())
        cost = compute_cost(X, y, w_current)
        cost_history.append(cost)
        if cost == 0:
            break
        
    return w_current, weights_history, cost_history


def stochastic_gradient_descent(X, y, w, learning_rate, num_iters, batch_size):
    """
     Performs stochastic gradient descent optimization

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :param learning_rate: learning rate
    :type learning_rate: float
    :param num_iters: number of iterations
    :type num_iters: int
    :param batch_size: size of the minibatch
    :type batch_size: int
    :return: weights, weights history, cost history
    :rtype: np.array(shape=(d, 1)), list, list
    """
    N = X.shape[0]
    weights_history = [w.flatten()]
    cost_history = [compute_cost(X, y, w)]
    
    w_current = w
    for t in range(0, num_iters-1):
        idx = np.random.choice(N, batch_size, replace=False)
        X_batch = X[idx, :]
        y_batch = y[idx,:]
        grad = compute_wgrad(X_batch, y_batch, w_current)
        w_current = w_current - learning_rate * grad
        weights_history.append(w_current.flatten())
        cost = compute_cost(X_batch, y_batch, w_current)
        cost_history.append(cost)
        if cost == 0:
            break
        
    return w_current, weights_history, cost_history

