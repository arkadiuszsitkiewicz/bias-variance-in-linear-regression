from src.calc.linear_reg import LinearRegression
import numpy as np


class LearningCurve:
    def __init__(self, X, y, X_val, y_val, lamb, proceed : bool = True):
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.lamb = lamb
        self.error_train = 0
        self.error_val = 0
        if proceed:
            self.__learn_curve()

    def __learn_curve(self):
        m = len(self.X)
        if np.mean(self.X[:, 0:1]) == 1:
            X_cost = self.X
            X_val_cost = self.X_val
        else:
            X_cost = LinearRegression.x_for_model_train(self.X)
            X_val_cost = LinearRegression.x_for_model_train(self.X_val)
        # y = self.y

        error_train = np.zeros((m, 2))
        error_val = np.zeros((m, 2))

        for i in range(1, m+1):
            # temp_err_train = 0
            # temp_err_val = 0
            theta = LinearRegression(X_cost[:i], self.y[:i], self.lamb).optimal_thetas
            error_train[i - 1] = i, LinearRegression.cost(theta, X_cost[:i, :], self.y[:i, :], 0)
            error_val[i - 1] = i, LinearRegression.cost(theta, X_val_cost, self.y_val, 0)


            # for j in range(10):
            #     randomness = np.random.choice(range(m), m, replace=False)
            #     X_cost = X_cost[randomness]
            #     y = y[randomness]
            #
            #     theta = LinearRegression(X_cost[:i], y[:i], self.lamb).optimal_thetas
            #     #  = i, LinearRegression.cost(theta, X_cost[:i, :], y[:i, :], 0)
            #     # error_val[i-1] = i, LinearRegression.cost(theta, X_val_cost, self.y_val, 0)
            #     temp_err_train += LinearRegression.cost(theta, X_cost[:i, :], y[:i, :], 0)
            #     temp_err_val += LinearRegression.cost(theta, X_val_cost, self.y_val, 0)
            #
            # error_train[i - 1] = i , temp_err_train/10
            # error_val[i - 1] = i,temp_err_val/10

        self.error_train = error_train
        self.error_val = error_val

