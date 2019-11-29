from multiAgents import *

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, currGameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    "*** YOUR CODE HERE ***"
    def miniMax(gameState, depth, alpha, beta, agent):
      if depth == 0 or gameState.isLose() or gameState.isWin():
        return self.evaluationFunction(gameState)
      # 0 for pacman -> maximizing
      if agent == 0:
        value = -float('inf')
        actions = gameState.getLegalActions(0)
        for action in actions:
          value = max(value, miniMax(gameState.generateSuccessor(0, action), depth - 1, alpha, beta, 1))
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
        value = float('inf')
        actions = gameState.getLegalActions(agent)
        for action in actions:
          value = min(value, miniMax(gameState.generateSuccessor(agent, action), depth, alpha, beta, nextAgent))
          beta = min(beta, value)
          if beta <= alpha:
            break
        return value

    legalActions = currGameState.getLegalActions(0)
    legalActions.remove(Directions.STOP)
    bestAction = legalActions[0]

    valueOfAction = -float('inf')
    for action in legalActions:
      value = miniMax(currGameState.generateSuccessor(0, action), self.depth, -float('inf'),float('inf'), 1)
      if value > valueOfAction: # Here we can see that value == value of action a lot
        bestAction = action # I need to change the evaluation function to change that
        valueOfAction = value

    return  bestAction
