import numpy as np


class PolyFeatures:

    @staticmethod
    def prepare_poly_features(x, p):
        m = len(x)
        x_poly = np.zeros((m, p))
        x_poly[:, 0:1] = x
        for i in range(1, m+1):
            x_poly[:, i:i+1] = x_poly[:, i-1:i]*x

        return x_poly

    @staticmethod
    def get_mean_sigma(x, p):
        x = PolyFeatures.prepare_poly_features(x, p)
        mu = np.mean(x, 0)
        sigma = np.std((x-mu), 0)
        return mu, sigma

    @staticmethod
    def feature_normalize(x, mean, sigma):
        return (x-mean)/sigma

    @staticmethod
    def final_poly_features(x, p, mean, sigma):
        m = len(x)
        x = PolyFeatures.prepare_poly_features(x, p)
        x = PolyFeatures.feature_normalize(x, mean, sigma)
        return np.c_[np.ones((m, 1)), x]

    def __init__(self, p, X, X_val, X_test):
        mean, sigma = PolyFeatures.get_mean_sigma(X, p)
        self.X = PolyFeatures.final_poly_features(X, p, mean, sigma)
        self.X_val = PolyFeatures.final_poly_features(X_val, p, mean, sigma)
        self.X_test = PolyFeatures.final_poly_features(X_test, p, mean, sigma)


if __name__== "__main__":
    x = np.array([1,2, 3, 4, 5]).reshape(-1, 1)

    print(type(x))
    x = PolyFeatures.prepare_poly_features(x, 3)
    print(x)