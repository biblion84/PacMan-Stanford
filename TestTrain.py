import pandas as pd


path = "/Users/macbookpro/Desktop/Ynov/A.I./pacman/PacMan-Stanford/dataGameWonMoreThan1500Map.csv"

data = pd.read_csv(path)

print(data.shape)

print(data.describe())

data.drop([0], axis=0)

print(data.shape)

print(data.head(5))