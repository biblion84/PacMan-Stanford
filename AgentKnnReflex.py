import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from joblib import dump, load


data = pd.read_csv("ReflexMatrix.csv")
y = data["Action"]
X = data.drop(columns=["Action"], axis=1)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8)
knn = neighbors.KNeighborsClassifier(n_neighbors=6)
knn.fit(xtrain, ytrain)

dump(knn, 'TrainedModels/knn.joblib', protocol=2)