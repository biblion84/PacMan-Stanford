import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors

path = "/Users/macbookpro/Desktop/Ynov/A.I./pacman/PacMan-Stanford/dataGameWonMoreThan1500WithColumnNames.csv"

dataTrain = pd.read_csv(path)
print(dataTrain.head())
dataTrain = dataTrain.drop(["lastAction"], axis=1)
dataTrain = dataTrain.drop(["Unnamed: 0"], axis=1)
dataTarget = dataTrain["labelNextAction"]
dataTrain = dataTrain.drop(columns=["labelNextAction"], axis=1)



print(dataTrain.shape)
print(dataTarget.shape)

print(dataTarget.head())
print(dataTrain.head())

xtrain, xtest, ytrain, ytest = train_test_split(dataTrain, dataTarget, train_size=0.8)

knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(xtrain, ytrain)

test = knn.predict(xtrain)
print(knn.predict(xtrain))

# errors = []
# for k in range(1,15):
#     knn = neighbors.KNeighborsClassifier(k)
#     errors.append(100*(1 - knn.fit(xtrain, ytrain).score(xtest, ytest)))
# plt.plot(range(1,15), errors, 'o-')
# plt.show()

error = 1 - knn.score(xtest, ytest)
print(knn.score(xtest, ytest))
print('Erreur: %f' % error)
