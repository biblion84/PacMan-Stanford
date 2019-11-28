import pandas as pd


path = "/Users/macbookpro/Desktop/Ynov/A.I./pacman/PacMan-Stanford/dataGameWonMoreThan1300WithColumnNames.csv"

dataTrain = pd.read_csv(path)

dataTest = dataTrain.drop("labelNextAction", axis=1)
dataTrain = dataTrain.drop([1], axis=1)
dataTest = dataTest.drop([1], axis=1)

print(dataTrain.shape)
#dataTrain = dataTrain.drop([0], axis=0)
print(dataTest.shape)
#dataTest = dataTest.drop([0], axis=0)


#print(dataTrain.shape)
#print(dataTest.shape)
#dataTrain.ghostUp = dataTrain.ghostUp.astype(float)
#print(type(dataTrain.ghostUp[1]))
print(dataTrain.head())
print(dataTest.head())

#dataTest.to_csv("PacmanTest.csv")
#dataTrain.to_csv("PacmanTrain.csv")