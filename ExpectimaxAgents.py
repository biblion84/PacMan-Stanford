from multiAgents import *

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """
  
  def getAction(self, currGameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    
    def expectiMax(gameState, depth, alpha, beta, agent):
      if depth == 0 or gameState.isLose() or gameState.isWin():
        return self.evaluationFunction(gameState)
      # 0 for pacman -> maximizing
      if agent == 0:
        value = -float('inf')
        actions = gameState.getLegalActions(0)
        for action in actions:
          value = max(value, expectiMax(gameState.generateSuccessor(0, action), depth - 1, alpha, beta, 1))
          alpha = max(alpha, value)
          if beta <= alpha:
            break
        return value
      # If not 0 it's a ghost baby -> minimizing
      # And now i need to minimize multiple times
      else:
        nextAgent = agent + 1
        if nextAgent == gameState.getNumAgents():
          nextAgent = 0
          depth = depth - 1
        # value = float('inf')
        actions = gameState.getLegalActions(agent)
        values = []
        for action in actions:
          values.append(expectiMax(gameState.generateSuccessor(agent, action), depth, alpha, beta, nextAgent))
        return (sum(values) / len(values))
        beta = min(beta, value)
        return value
    
    legalActions = currGameState.getLegalActions(0)
    legalActions.remove(Directions.STOP)
    bestAction = legalActions[0]
    valueOfAction = -float('inf')
    nextGameState = False
    for action in legalActions:
      nextGS = currGameState.generateSuccessor(0, action)
      value = expectiMax(nextGS, self.depth, -float('inf'), float('inf'), 1)
      value -= AlreadyVisitedScore(nextGS.getPacmanPosition()) * 0.1
      if value > valueOfAction:  # Here we can see that value == value of action a lot
        bestAction = action  # I need to change the evaluation function to change that
        valueOfAction = value
        nextGameState = currGameState.generateSuccessor(0, action)
    
    HistoActions.append(bestAction)
    AlreadyVisited.append(nextGameState.getPacmanPosition())
    self.dataFrame.loc[self.indexDataframe, :] = extractFeature(currGameState, bestAction)
    self.indexDataframe = self.indexDataframe + 1
    
    if (nextGameState.isWin()):
      if (nextGameState.getScore() > 1500):
        with open('dataGameWonMoreThan1500WithColumnNames.csv', 'a') as f:
          # self.dataFrame.columns = ["ghostUp","ghostDown","ghostLeft","ghostRight","wallUp","wallDown","wallLeft","wallRight","foodUp","foodDown","foodLeft","foodRight","emptyUp","emptyDown","emptyLeft","emptyRight","nearestFood","nearestGhost","nearestCapsule","legalPositionUp","legalPositionDown","legalPositionULeft","legalPositionRight","pacmanPositionX","pacmanPositionY","labelNextAction"]
          
          if (self.alreadyWroteHeaders):
            self.dataFrame.to_csv(f, header=False)
          else:
            self.dataFrame.to_csv(f, header=True)
            self.alreadyWroteHeaders = True
    return bestAction


