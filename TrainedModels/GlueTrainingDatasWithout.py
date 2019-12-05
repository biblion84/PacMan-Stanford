import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from joblib import dump, load
from sklearn import tree
from sklearn.model_selection import GridSearchCV
datas = []

# data = pd.read_csv("ReflexMatrix3.csv")
# data1 = pd.read_csv("ReflexMatrix4.csv")
# data2 = pd.read_csv("ReflexMatrix5.csv")

# datas.append(data)
# datas.append(data1)
# datas.append(data2)


for i in range(0,7):
    data = pd.read_csv("DistanceReflexMatrixbatchWithout" + str(i) + ".csv")
    datas.append(data)

glued = pd.concat(datas)

with open("DistanceGluedWithout2.csv", 'a') as f:
    # self.dataFrame.columns = ["ghostUp","ghostDown","ghostLeft","ghostRight","wallUp","wallDown","wallLeft","wallRight","foodUp","foodDown","foodLeft","foodRight","emptyUp","emptyDown","emptyLeft","emptyRight","nearestFood","nearestGhost","nearestCapsule","legalPositionUp","legalPositionDown","legalPositionULeft","legalPositionRight","pacmanPositionX","pacmanPositionY","labelNextAction"]
    glued.to_csv(f, header=True, index=False)
