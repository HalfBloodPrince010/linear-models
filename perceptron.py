import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


class Perceptron:
    def __init__(self, data_values, labels, n, learning_rate, dimension, iterations):
        self.data = data_values
        self.label = labels
        self.dataset_size = n
        self.learning_rate = learning_rate
        self.weights = np.random.rand(1, dimension+1)
        # self.weights = np.zeros((1, dimension+1))
        self.misclassified = []
        self.iterations = iterations

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
            # while there is some mis-classifed point
            if summation != self.label[i]:  # y=-1, signal=1 or y=1, signal=-1
                self.misclassified.append(i)

        converged = False
        iteration = 0
        while not converged and iteration < self.iterations:
            if len(self.misclassified) == 0:
                converged = True
                continue
            else:
                print("==================================================")
                print("Initial misclassification", len(self.misclassified))
                random_index = random.randint(0, len(self.misclassified)-1)
                misclassified_constraint_index = self.misclassified[random_index]
                print(misclassified_constraint_index)
                print(self.data[misclassified_constraint_index, :])
                # Now fetch that data point from numpy set and update it with weight
                # self.weights = self.weights + (self.learning_rate * label[misclassified_constraint_index]*self.data[misclassified_constraint_index])
                if label[misclassified_constraint_index] == -1:  # y=0, signal=1
                    self.weights = self.weights - (self.learning_rate * self.data[misclassified_constraint_index])
                elif label[misclassified_constraint_index] == 1:
                    self.weights = self.weights + (self.learning_rate * self.data[misclassified_constraint_index])

                self.misclassified = []

                for i in range(self.dataset_size):
                    signal = np.dot(self.data[i, :], self.weights.T)
                    # while there is some mis-classifed point
                    if signal > 0:
                        summation = 1
                    else:
                        summation = -1
                    # while there is some mis-classifed point
                    if summation != self.label[i]:  # y=-1, signal=1
                        self.misclassified.append(i)
                print("Later: ", len(self.misclassified))

                iteration += 1

        print(self.weights)


if __name__ == "__main__":
    # Process the Data
    print("Processing the data")
    dataframe = pd.read_csv("classification.txt", names=["x", "y", "z", "label", "label2"])
    dataf = dataframe.iloc[:, 0:4]
    data = dataf[['x', 'y', 'z']]
    label = dataf[['label']].to_numpy()
    d = pd.DataFrame(np.ones((len(dataframe), 1)))
    data.insert(0, 'x0', d)
    data = data.to_numpy()
    x = Perceptron(data, label, len(data), 0.5, 3, 100)
    x.train()



