# version 1.1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from RedWineClassifier.neural_net import NeuralNetwork
from RedWineClassifier.operations import *

def load_dataset(csv_path, target_feature):
    dataset = pd.read_csv(csv_path)
    t = np.expand_dims(dataset[target_feature].to_numpy().astype(float), axis=1)
    X = dataset.drop([target_feature], axis=1).to_numpy()
    return X, t

X, y = load_dataset("data/wine_quality.csv", "quality")

n_features = X.shape[1]
net = NeuralNetwork(n_features, [8,8,4,1], [ReLU(), ReLU(), Sigmoid(), Identity()], MeanSquaredError(), learning_rate=0.001)
epochs = 500
k = 5

kf = KFold(n_splits=k)

errors = []
epoch_losses_mean = [0 for _ in range(epochs)]

for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    trained_W, epoch_losses = net.train(X_train, y_train, epochs)
    
    for index, loss in enumerate(epoch_losses):
        epoch_losses_mean[index] += loss

    errors.append(net.evaluate(X_test, y_test, mean_absolute_error))
    print("Error on test set for fold {}: {}\n".format(i, errors[-1]))

for index, loss in enumerate(epoch_losses_mean):
    epoch_losses_mean[index] /= k

print(f"\n\nAverage of MAE: {round(np.mean(errors), 4)}")
print(f"Standard Deviation of MAE: {round(np.std(errors), 4)}")

plt.xlabel('Epoch Number')
plt.ylabel('Average Training Loss')

plt.plot(np.arange(0, epochs), epoch_losses_mean)
plt.xticks(np.arange(0, epochs, 1))
plt.show()