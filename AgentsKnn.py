import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn import neighbors

path = "/Users/macbookpro/Desktop/Ynov/A.I./pacman/PacMan-Stanford/"



dataTrain = pd.read_csv(path + "PacmanTrain.csv")
dataTrain = pd.get_dummies(dataTrain)
dataTrain = dataTrain.drop("Unnamed: 0", axis=1)
dataTrain = dataTrain.drop("Unnamed: 0.1", axis=1)
print(dataTrain.dtypes)
dataTest = pd.read_csv(path + "PacmanTest.csv")
dataTest = pd.get_dummies(dataTest)
dataTest = dataTest.drop("Unnamed: 0", axis=1)
dataTrain = dataTrain.drop("Unnamed: 0.1", axis=1)

xtrain, xtest, ytrain, ytest = train_test_split(dataTrain, dataTest, train_size=0.8)

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)

error = 1 - knn.score(xtest, ytest)
print('Erreur: %f' % error)
