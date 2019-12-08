import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from joblib import dump, load
from sklearn import tree
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("VoteGlued.csv")
y = data["labelNextAction"]
X = data.drop(columns=["labelNextAction"], axis=1)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.9)
# # #
parameters = {'criterion':['gini','entropy'],'max_depth':[50, 70, 100, 120, 150]}
gs = GridSearchCV(tree.DecisionTreeClassifier(), parameters, cv=3)

gs = gs.fit(xtrain, ytrain)
print gs.best_params_
print gs.best_score_
dump(gs, 'TrainedModels/gsPlus.joblib', protocol=2)

clf = tree.DecisionTreeClassifier(criterion=gs.best_params_["criterion"], max_depth=gs.best_params_["max_depth"])

clf = clf.fit(xtrain, ytrain)
dump(clf, 'TrainedModels/clfMatrixPlus.joblib', protocol=2)

# #
# clf = load('TrainedModels/clfMatrixPlus.joblib')
# data = pd.read_csv("DistanceGlued2.csv")
# y = data["labelNextAction"]
# X = data.drop(columns=["labelNextAction"], axis=1)

# print clf.score(X,y)

# xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.5)
