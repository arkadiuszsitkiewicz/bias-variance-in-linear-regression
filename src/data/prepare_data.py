import numpy as np


class PrepareData:

    def __init__(self,raw_data):
        self.__raw_data = raw_data
        self.__data_as_matrix = self.__csv_to_matrix()
        self.X = self.__data_as_matrix[0:12, 0:1]
        self.y = self.__data_as_matrix[0:12, 1:2]
        self.X_val = self.__data_as_matrix[12:33, 0:1]
        self.y_val = self.__data_as_matrix[12:33, 1:2]
        self.X_test = self.__data_as_matrix[33:, 0:1]
        self.y_test = self.__data_as_matrix[33:, 1:2]

    def __csv_to_matrix(self):
        return np.loadtxt(open(self.__raw_data,"rb"),delimiter=';', dtype="float")
