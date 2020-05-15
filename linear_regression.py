import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class LinearRegression:
    def __init__(self, input_matrix, y_vector, n):
        self.data = input_matrix
        self.outputs = y_vector
        self.n = n
        self.weights = None

    def train(self):
        pseudo_inverse = ((self.data.T * self.data).I * self.data.T)
        self.weights = np.dot(pseudo_inverse, self.outputs)
        print("=============================================")
        self.weights = np.squeeze(np.asarray(self.weights))
        print("Weights", self.weights)
        print("=============================================")
        # Finding weight by using pseudo inverse

    def predict(self, test):
        prediction = np.dot(self.weights.T, test)
        print(prediction)


if __name__ == "__main__":
    # Process the Data
    dataframe = pd.read_csv("linear_regression.txt", delimiter=",", names=['x1', 'x2', 'y'])
    datapd = dataframe[['x1', 'x2']]
    label = dataframe['y']
    d = pd.DataFrame(np.ones((len(dataframe), 1)))
    datapd.insert(0, 'x0', d)
    data = datapd.to_numpy()
    data = np.asmatrix(data)
    x = LinearRegression(data, label, len(dataframe))
    x.train()
    """
    Another way of Calculating
    
    m = np.linalg.inv(np.dot(data.T, data))
    n = m*data.T
    print("n\n", n)
    w = np.dot(n, label)
    print(w)
    
    """

