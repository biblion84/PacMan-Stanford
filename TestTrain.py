import pandas as pd


path = "/Users/macbookpro/Desktop/Ynov/A.I./pacman/PacMan-Stanford/dataGameWonMoreThan1500WithColumnNames.csv"

dataTrain = pd.read_csv(path)

dataTarget = dataTrain["labelNextAction"]

print(dataTrain.shape)
#dataTrain = dataTrain.drop([0], axis=0)
print(dataTarget.shape)
#dataTest = dataTest.drop([0], axis=0)


#print(dataTrain.shape)
#print(dataTest.shape)
#dataTrain.ghostUp = dataTrain.ghostUp.astype(float)
#print(type(dataTrain.ghostUp[1]))
print(dataTrain.head())
print(dataTarget.head())

print(dataTrain.shape)
print(dataTarget.shape)

dataTarget.to_csv("PacmanTarget.csv")
dataTrain.to_csv("PacmanTrain.csv")

print(dataTrain.shape)
print(dataTarget.shape)