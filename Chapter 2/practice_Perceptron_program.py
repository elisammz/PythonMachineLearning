import numpy as np


class Perceptron(object):
    """
    PERCEPTRON CLASSIFIER

    Parameters:
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_inter: int
        Passes over the training dataset.

    Attributes:
    w_: 1d-array
        Weights after fitting
    errors_: list
        Number of misclassifications in every epoch

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    # Fiting object
    def fit(self, X, y):
        """
        FIT TRAINING DATA

        Parameters:
        X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples es is the number of samples and n_features is the number of features.

        y: array-like, shape = [n_samples]
        Target values

        Returns:
        self: object

        """

        self.w_ = np.zeros(
            1 + X.shape[1]
        )  # 1d-array is fill in with zeros and it has the same number of rows of the array with the samples + 1
        self.errors_ = []  # new array for errors

        #  Since we are coding for the training data, for each training sample xi performs the following steps:
        # 1. Compute the output value ^y. The output value is the class label predicted (1, -1) by the unit step function (ø(z)). Page 19
        # 2. Update the weights

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # Delta W is update, and it is calculated by the equation 2 in page 22, where self.eta is the is n(learning rate), target is the true class of the xi in the training sample, and self.predict(xi) is the predicted class label
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                # update is saved in the first element of the self.w_ array
                self.w_[0] += update
                errors += int(update != 0.0)  # number of misclassifications
            self.errors_.append(errors)
        return self

    # Net input method which is later passed to the activation function (here, ø with labels of 1 or -1)
    def net_input(self, X):
        """Calculate net input"""
        # Calculation of the vector dot product w^T*x
        # returns the dot product of X and self.w_, and then sums the values of the update
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after the unit step (or activation, ø) function"""
        # just as reminder, the activation function in this program is the unit step function or ø with labels 1, -1
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    # reminder for future step:
    """epochs: the number of passes over the training data set. Epochs are set by modifying the eta and n_iter"""


"""Implementation of the Ov A technique to obtaine the class label that is closer to the largest absolute net input valie"""
