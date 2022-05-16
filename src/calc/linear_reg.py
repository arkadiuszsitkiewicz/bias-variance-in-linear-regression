import numpy as np
import scipy.optimize as op


class LinearRegression:

    @staticmethod
    def cost(theta, X, y, lamb):
        m, n = X.shape
        theta = theta.reshape((n, 1))
        h = X.dot(theta).reshape((m, 1))
        return 1/(2*m)*np.sum((h-y)**2) + lamb/(2*m)*sum((theta[1:]**2))

    @staticmethod
    def gradient(theta, X, y, lamb):
        m, n = X.shape
        theta = theta.reshape((n, 1))
        h = X.dot(theta)
        grad = 1/m*(X.T).dot(h-y)
        grad[1:, :] += lamb/m*(theta[1:]**2)
        return grad.flatten()

    @staticmethod
    def x_for_model_train(X):
        m = len(X)
        return np.c_[np.ones((m, 1)), X]

    def __init__(self, X: np.ndarray, y: np.ndarray, lamb: float):
        if np.mean(X[:, 0] == 1):
            self.X = X
        else:
            self.X = LinearRegression.x_for_model_train(X)
        self.y = y
        self.lamb = lamb
        self.optimal_thetas = self.__train_linear_reg()

    def __train_linear_reg(self) -> np.ndarray:

        X = self.X
        y = self.y
        lamb = self.lamb
        m, n = X.shape
        initial_theta = np.zeros((n, 1))

        theta_min = op.minimize(fun=LinearRegression.cost, x0=initial_theta, args=(X, y, lamb),\
                                method="Newton-CG", jac=LinearRegression.gradient)
        return theta_min.x











