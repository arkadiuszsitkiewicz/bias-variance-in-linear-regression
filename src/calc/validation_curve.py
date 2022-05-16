from src.calc.linear_reg import LinearRegression
import numpy as np

class ValidationCurve:

    def __init__(self, X, y, X_val, y_val, proceed : bool = True):
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.error_train = None
        self.error_val = None
        if proceed:
            self.__lambda_dependence()

    def __lambda_dependence(self,):

        if np.mean(self.X[:,0:1]) == 1:
            X_cost = self.X
            X_val_cost = self.X_val
        else:
            X_cost = LinearRegression.x_for_model_train(self.X)
            X_val_cost = LinearRegression.x_for_model_train(self.X_val)

        lamb_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
        m = len(lamb_vec)
        error_train = np.zeros((m, 2))
        error_val = np.zeros((m, 2))

        for i in range(m):
            lamb = lamb_vec[i]
            theta = LinearRegression(X_cost , self.y, lamb).optimal_thetas

            error_train[i] = lamb, LinearRegression.cost(theta, X_cost, self.y, 0)
            error_val[i] = lamb, LinearRegression.cost(theta, X_val_cost, self.y_val, 0)

        self.error_train = error_train
        self.error_val = error_val

