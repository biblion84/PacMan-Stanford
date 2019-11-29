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
import keras
from keras import losses
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


class KerasAgent(MultiAgentSearchAgent):
  model = None

  def __init__(self):
    model = Sequential()
    model.add(Dense(units=128, activation='softmax', input_dim=26))
    model.add(Dense(units=128, activation='relu'))
    # model.add(Dense(units=128, activation='relu'))
    # model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=4, activation='relu'))
    # model.summary()
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
    self.dataTrain = pd.read_csv("dataGameWonMoreThan1500WithColumnNames.csv")
    # self.dataTrain = self.dataTrain.drop(self.dataTrain.columns[0], axis=1)
    self.dataTarget = self.dataTrain["labelNextAction"]
    self.dataTrain = self.dataTrain.drop(columns=["labelNextAction"], axis=1)
    self.dataTarget = to_categorical(self.dataTarget)
    model.fit(self.dataTrain, self.dataTarget, epochs=3, batch_size=128)
    loss_and_metrics = model.evaluate(self.dataTrain, self.dataTarget)
    self.model = model
    print(loss_and_metrics)

  def getAction(self, currGameState):
    data = pd.DataFrame(columns=dataColumns)
    data.loc[0, :] = extractFeature(currGameState, "South")
    dataTrain = data.drop(columns=["labelNextAction"], axis=1)
    nextActionNumber = self.model.predict_classes(dataTrain)
    print (nextActionNumber)
    nextPredictedAction = getActionByNumber(nextActionNumber)

    if (nextPredictedAction not in currGameState.getLegalActions(0)):
      print("Illegal Action")
      print(nextPredictedAction)
      nextPredictedAction = currGameState.getLegalActions(0)[random.randrange(0, len(currGameState.getLegalActions(0)))]

    return nextPredictedAction