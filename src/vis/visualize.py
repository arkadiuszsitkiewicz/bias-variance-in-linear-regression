from src.calc.poly_features import PolyFeatures
import matplotlib.pyplot as plt
import numpy as np


class Visualization:

    VIS_TYPE = ("DATA", "FIT", "LEARNING CURVE", "SET LAMBDA")

    @classmethod
    def getvistype(cls):
        return cls.VIS_TYPE

    def __init__(self, X, y, theta: np.ndarray = False, degree: int = False, lamb: float = False, vistype="DATA"):
        self.X = X
        self.y = y
        vistype = vistype.upper().strip()

        if vistype not in Visualization.VIS_TYPE:
            raise ValueError(f"{vistype} is not a valid visualization type")
        elif vistype == "DATA":
            self.__plot_data(False)
        elif vistype == "FIT":
            if type(theta) == bool or type(degree) == bool or type(lamb) == bool:
                raise ValueError(f"Provide correct datatype for theta, degree and lamb parameters")
            else:
                self.theta = theta
                self.degree = degree
                self.lamb = lamb
                self.__plot_data(True)
        else:
            if type(degree) == bool or type(lamb) == bool:
                raise ValueError(f"Provide correct datatype for degree and lamb parameters")
            else:
                self.degree = degree
                self.lamb = lamb
                self.__plot_learning_curve(vistype)

    def __plot_data(self, fit):
        plt.figure(figsize=(7, 7))
        plt.plot(self.X, self.y, "b+", markersize=7, label="data points")
        plt.xlabel("Change in water level (x)")
        plt.ylabel("Water flowing out of the dam (y)")

        if fit:
            x_min = int(np.min(self.X)) - 10
            x_max = int(np.max(self.X)) + 10
            if len(self.theta) > 2:
                x_min -= 40
                x_max += 40

            x_fit = np.arange(x_min, x_max, 0.1).reshape((-1, 1))
            m, n = x_fit.shape

            if len(self.theta) > 2:
                mean, sigma = PolyFeatures.get_mean_sigma(self.X, self.degree)
                x_fit_poly = PolyFeatures.final_poly_features(x_fit, self.degree, mean, sigma)
                y_fit = x_fit_poly.dot(self.theta)
            else:
                y_fit = np.c_[np.ones((m, 1)), x_fit].dot(self.theta)
            plt.plot(x_fit, y_fit, "r-", markersize=1, label="Hypothesis fit")
            plt.title(f"Hypothesis fit with degree polynomial {self.degree} and lambda = {self.lamb}", fontweight="bold")

        plt.ylim([np.min(self.y)-10, np.max(self.y)+10])
        plt.legend()
        plt.show()

    def __plot_learning_curve(self, vistype):
        plt.figure(figsize=(7, 7))

        if vistype == "LEARNING CURVE":
            title = f"Learning curve for linear regression\nDegree polynomial: {self.degree} & lambda: {self.lamb}"
            x_label = "Number of training examples"
        else:
            title = "Validation curve - lambda dependency"
            x_label = "Lambda"
            plt.xlim([0, 10])
            plt.ylim([0, 25])

        plt.title(title, fontweight="bold", loc="left")
        plt.plot(self.X[:, 0],self.X[:, 1], label="Train")
        plt.plot(self.y[:, 0], self.y[:, 1], color="red", label="Cross Validation")
        plt.xlabel(x_label)
        plt.ylabel("Error")
        plt.legend()
        # plt.ylim([0, np.max(self.y)*3/4])

        plt.show()
