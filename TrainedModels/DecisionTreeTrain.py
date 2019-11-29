import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from joblib import dump, load
from sklearn import tree
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("ReflexMatrix.csv")
y = data["labelNextAction"]
X = data.drop(columns=["labelNextAction"], axis=1)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8)





parameters = {'criterion':['gini','entropy'],'max_depth':[0.1, 1, 10, 100, 1000, 10000]}
gs = GridSearchCV(tree.DecisionTreeClassifier(), parameters, cv=5)

gs = gs.fit(xtrain, ytrain)
print gs.best_params_
print gs.best_score_
clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=10)

clf = clf.fit(xtrain, ytrain)
dump(clf, 'TrainedModels/clfMatrix.joblib', protocol=2)