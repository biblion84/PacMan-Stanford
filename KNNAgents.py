import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from multiAgents import MultiAgentSearchAgent, extractFeature, getActionByNumber

class KNNAgent(MultiAgentSearchAgent):
  def __init__(self):
    self.dataTrain = pd.read_csv("dataGameWonMoreThan1500WithColumnNames.csv")
    self.dataTrain = self.dataTrain.drop(self.dataTrain.columns[0], axis=1)
    self.dataTarget = self.dataTrain["labelNextAction"]
    self.dataTrain = self.dataTrain.drop(columns=["labelNextAction"], axis=1)
    xtrain, xtest, ytrain, ytest = train_test_split(self.dataTrain, self.dataTarget, train_size=0.8)
    self.knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    self.knn.fit(xtrain, ytrain)
  
  def getAction(self, currGameState):
    data = pd.DataFrame(
      columns=["ghostUp", "ghostDown", "ghostLeft", "ghostRight", "wallUp", "wallDown", "wallLeft", "wallRight",
               "foodUp", "foodDown", "foodLeft", "foodRight", "emptyUp", "emptyDown", "emptyLeft", "emptyRight",
               "nearestFood", "nearestGhost", "nearestCapsule", "legalPositionUp", "legalPositionDown",
               "legalPositionULeft", "legalPositionRight", "pacmanPositionX", "pacmanPositionY", "labelNextAction"])
    data.loc[0, :] = extractFeature(currGameState, "South")
    dataTrain = data.drop(columns=["labelNextAction"], axis=1)
    nextActionNumber = self.knn.predict(dataTrain)
    
    nextPredictedAction = getActionByNumber(nextActionNumber)
    
    if (nextPredictedAction not in currGameState.getLegalActions(0)):
      print("Illegal Action")
      nextPredictedAction = currGameState.getLegalActions(0)[random.randrange(0, len(currGameState.getLegalActions(0)))]
    
    return nextPredictedAction