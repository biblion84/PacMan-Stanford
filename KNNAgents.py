from multiAgents import *
from sklearn.model_selection import train_test_split
from sklearn import neighbors

class KNNAgent(MultiAgentSearchAgent):
  def __init__(self):
    self.dataTrain = pd.read_csv("dataGameWonMoreThan1500WithColumnNames.csv")
    self.dataTarget = self.dataTrain["labelNextAction"]
    self.dataTrain = self.dataTrain.drop(columns=["labelNextAction"], axis=1)
    xtrain, xtest, ytrain, ytest = train_test_split(self.dataTrain, self.dataTarget, train_size=0.8)
    self.knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    self.knn.fit(xtrain, ytrain)
  
  def getAction(self, currGameState):
    data = pd.DataFrame(
      columns=dataColumns)
    data.loc[0, :] = extractFeature(currGameState, "South")
    dataTrain = data.drop(columns=["labelNextAction"], axis=1)
    nextActionNumber = self.knn.predict(dataTrain)
    
    nextPredictedAction = getActionByNumber(nextActionNumber)
    
    if (nextPredictedAction not in currGameState.getLegalActions(0)):
      print("Illegal Action")
      nextPredictedAction = currGameState.getLegalActions(0)[random.randrange(0, len(currGameState.getLegalActions(0)))]
    
    return nextPredictedAction