from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from joblib import dump, load
from sklearn import tree
from sklearn.model_selection import GridSearchCV

# data = pd.read_csv("DodgeGlued.csv")
# y = data["labelNextAction"]
# X = data.drop(columns=["labelNextAction"], axis=1)
# xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8)
# #
# parameters = {'criterion':['gini','entropy'],'max_depth':[1, 10, 100, 1000]}
# gs = GridSearchCV(tree.DecisionTreeClassifier(), parameters, cv=3)
#
# gs = gs.fit(xtrain, ytrain)
# print gs.best_params_
# print gs.best_score_
# dump(gs, 'TrainedModels/gs.joblib', protocol=2)

# clf = MLPClassifier()
#
# clf = clf.fit(xtrain, ytrain)
# dump(clf, 'TrainedModels/neuralNetwork.joblib', protocol=2)

#
clf = load('TrainedModels/neuralNetwork.joblib')
data = pd.read_csv("TrainedModels/DodgeGlued2.csv")
y = data["labelNextAction"]
X = data.drop(columns=["labelNextAction"], axis=1)

print clf.score(X,y)

# xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.5)
