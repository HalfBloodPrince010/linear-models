import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


class Pocket:
    def __init__(self, data_values, labels, n, learning_rate, dimension, iterations):
        self.data = data_values
        self.label = labels
        self.dataset_size = n
        self.learning_rate = learning_rate
        self.weights = np.random.rand(1, dimension+1)
        # self.weights = np.ones((1, dimension+1))
        self.minimum = float('inf')
        self.misclassified = []
        self.iterations = iterations
        self.best_weights = None

    def train(self):
        misclassified = []
        '''
        signal = wT.x(i) --> sign(signal)=output
        '''
        print(self.weights)
        for i in range(self.dataset_size):
            signal = np.dot(self.data[i, :], self.weights.T)
            if signal > 0:
                summation = 1
            else:
                summation = -1
            if summation != self.label[i]:  # y=-1, signal=1 or y=1, signal=-1
                self.misclassified.append(i)
                # break

        if len(self.misclassified) != 0:
            converged = False
            iteration = 0
            while not converged and (iteration < self.iterations):
                if len(self.misclassified) == 0:
                    converged = True
                    continue
                else:
                    random_index = random.randint(0, len(self.misclassified)-1)
                    misclassified_constraint_index = self.misclassified[random_index]
                    # print(self.data[misclassified_constraint_index, :])
                    # Now fetch that data point from numpy set and update it with weight
                    if label[misclassified_constraint_index] == -1:  # y=0, signal=1
                        self.weights = self.weights - (self.learning_rate * self.data[misclassified_constraint_index])
                    elif label[misclassified_constraint_index] == 1:
                        self.weights = self.weights + (self.learning_rate * self.data[misclassified_constraint_index])
                    predicted_array = self.prediction(self.weights)
                    misclassified = np.sum(predicted_array != self.label)
                    updated_misclassification = misclassified
                    if updated_misclassification < self.minimum:
                        self.minimum = updated_misclassification
                        self.best_weights = self.weights
                    iteration += 1
        else:
            self.best_weights = self.weights

        print("Best Weights:", self.best_weights, "\n Misclassifications:", self.minimum)

    def prediction(self, updated_weights):
        predicted_values = np.dot(self.data, updated_weights.T)
        predicted_values[np.where(predicted_values > 0)] = 1
        predicted_values[np.where(predicted_values <= 0)] = -1
        return predicted_values


if __name__ == "__main__":
    # Process the Data
    print("Processing the data")
    dataframe = pd.read_csv("classification.txt", names=["x", "y", "z", "label1", "label2"])
    dataf = dataframe.iloc[:, 0:4]
    data = dataframe[['x', 'y', 'z']]
    label = dataframe[['label2']].to_numpy()
    d = pd.DataFrame(np.ones((len(dataframe), 1)))
    data.insert(0, 'x0', d)
    data = data.to_numpy()
    x = Pocket(data, label, len(data), 0.05, 3, 7000)
    x.train()