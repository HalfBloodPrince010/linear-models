import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class LogisticRegression:
    def __init__(self, data_values, labels, n, learning_rate, dimension, t):
        self.data = data_values
        self.label = labels
        self.dataset_size = n
        self.learning_rate = learning_rate
        self.weights = np.random.rand(1, dimension+1)  # Random Weights initially
        self.iterations = t  # t corresponds to the iterations
        self.gradient = None
        self.final_weights = np.array([-0.031622, -0.17761997, 0.11452757, 0.07678213])

    def train(self):
        print("Data", self.data[0], "\n", self.label[0], "\n", self.dataset_size, "\n", self.weights)
        x = self.label[0]*np.dot(self.data[0], self.weights.T)
        print("Denominator", x)
        y = self.label[0]*self.data[0]
        print("Numerator", y)
        print(y/x)
        for i in range(self.iterations):
            weight_temp = np.zeros((1, 4))
            for j in range(self.dataset_size):
                denominator = 1 + 2.71828**(self.label[j]*np.dot(self.data[j], self.weights.T))
                numerator = self.label[j]*self.data[j]
                weight_temp += (numerator/denominator)
            self.gradient = (-1)*(weight_temp/self.dataset_size)
            self.weights = self.weights - (self.learning_rate*self.gradient)
        print("Final Weight", self.weights)
        # Final Weight [[-0.031622   -0.17761997  0.11452757  0.07678213]]
        self.final_weights = self.weights

    def prediction(self):
        classification = 0
        prediction = None
        for i in range(self.dataset_size):
            signal = np.dot(self.data[i], self.final_weights.T)
            x = 2.71828**signal
            theta = x/(1+x)
            if theta >= 0.5:
                prediction = 1
            else:
                prediction = (-1)
            if prediction == self.label[i]:
                classification += 1
        print(classification)


if __name__ == "__main__":
    # Process the Data
    print("Processing the data")
    dataframe = pd.read_csv("classification.txt", names=["x", "y", "z", "label1", "label2"])
    # dataf = dataframe.iloc[:, 0:4]
    data = dataframe[['x', 'y', 'z']]
    label = dataframe[['label2']].to_numpy()
    d = pd.DataFrame(np.ones((len(dataframe), 1)))
    data.insert(0, 'x0', d)
    data = data.to_numpy()
    # print(data)
    obj = LogisticRegression(data, label, len(data), 0.1, 3, 7000)
    obj.train()
    obj.prediction()
