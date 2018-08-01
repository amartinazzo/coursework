import numpy as np

def normal_equation_prediction(X, y):
    '''
    Calculates the prediction using the normal equation method.

    :param X: design matrix
    :type X: np.array
    :param y: regression targets
    :type y: np.array
    :return: prediction
    :rtype: np.array
    '''
    X_one = np.ones((X.shape[0], X.shape[1]+1))
    X_one[:,:-1] = X

    X_pseudo_inv = np.linalg.pinv(X_one)
    w = np.dot(X_pseudo_inv, y)
    y_hat = np.dot(X_one, w)
    
    return y_hat
