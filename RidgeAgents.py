from util import manhattanDistance, Queue
from game import Directions, Actions
import random, util
from collections import defaultdict
import math
from game import Agent
from MonteCarlo import MCTS, Node
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from multiAgents import MultiAgentSearchAgent, extractFeature, getActionByNumber, dataColumns

class RidgeAgent(MultiAgentSearchAgent):
  def __init__(self):
    self.dataTrain = pd.read_csv("dataGameWonMoreThan1500WithColumnNames.csv")
    self.dataTrain = self.dataTrain.drop(self.dataTrain.columns[0], axis=1)
    self.dataTarget = self.dataTrain["labelNextAction"]
    self.dataTrain = self.dataTrain.drop(columns=["labelNextAction"], axis=1)
    xtrain, xtest, ytrain, ytest = train_test_split(self.dataTrain, self.dataTarget, train_size=0.8)
    self.rr = Ridge(alpha=100)
    self.rr.fit(xtrain, ytrain)

  def getAction(self, currGameState):
    data = pd.DataFrame(
      columns=dataColumns)
    data.loc[0, :] = extractFeature(currGameState, "South")
    dataTrain = data.drop(columns=["labelNextAction"], axis=1)
    nextActionNumber = self.rr.predict(dataTrain)

    nextPredictedAction = getActionByNumber(nextActionNumber)

    if (nextPredictedAction not in currGameState.getLegalActions(0)):
      print("Illegal Action")
      nextPredictedAction = currGameState.getLegalActions(0)[random.randrange(0, len(currGameState.getLegalActions(0)))]

    return nextPredictedAction