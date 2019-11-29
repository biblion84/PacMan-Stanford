from multiAgents import *
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from joblib import dump, load

class KNNAgent(MultiAgentSearchAgent):
  def __init__(self):
    self.knn = load('TrainedModels/knn.joblib')
  
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
    return  nextPredictedAction
