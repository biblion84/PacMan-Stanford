import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors

path = "/Users/macbookpro/Desktop/Ynov/A.I./pacman/PacMan-Stanford/dataGameWonMoreThan1500WithColumnNames.csv"

dataTrain = pd.read_csv(path)
print(dataTrain.head())
dataTarget = dataTrain["labelNextAction"]
dataTrain = dataTrain.drop(columns=["labelNextAction"], axis=1)



print(dataTrain.shape)
print(dataTarget.shape)

print(dataTarget.head())
print(dataTrain.head())

xtrain, xtest, ytrain, ytest = train_test_split(dataTrain, dataTarget, train_size=0.8)

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)

error = 1 - knn.score(xtest, ytest)
print(knn.score(xtest, ytest))
print('Erreur: %f' % error)
