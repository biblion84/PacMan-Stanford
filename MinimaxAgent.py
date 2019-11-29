from multiAgents import *

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, currGameState):
    def miniMax(gameState, depth, agent):
      if depth == 0 or  gameState.isLose() or gameState.isWin():
        return self.evaluationFunction(gameState)
      #0 for pacman -> maximizing
      if agent == 0:
        value = -float('inf')
        actions = gameState.getLegalActions(0)
        for action in actions:
          value = max(value, miniMax(gameState.generateSuccessor(0, action), depth - 1, 1))
        return value
      # If not 0 it's a ghost baby -> minimizing
      # And now i need to minimize multiple times
      else:
        nextAgent = agent + 1
        if nextAgent == gameState.getNumAgents() :
          nextAgent = 0
          depth = depth -1
        value = float('inf')
        actions = gameState.getLegalActions(agent)
        for action in actions:
          value = min(value, miniMax(gameState.generateSuccessor(agent, action), depth, nextAgent))
        return value

    legalActions = currGameState.getLegalActions(0)
    legalActions.remove(Directions.STOP)
    bestAction = legalActions[0]
    valueOfAction = -float('inf')
    for action in legalActions:
      value = miniMax(currGameState.generateSuccessor(0, action), self.depth, 1)
      if value > valueOfAction: # Here we can see that value == value of action a lot
        bestAction = action # I need to change the evaluation function to change that
        valueOfAction = value

    return  bestAction

    # Algo taken from the pseudocode from minimax wikipedia
