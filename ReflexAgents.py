from multiAgents import *

class ReflexAgent(Agent):
  def getAction(self, gameState):
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currGameState, pacManAction):
    nextGameState = currGameState.generatePacmanSuccessor(pacManAction)
    oldPos = currGameState.getPacmanPosition()
    newPos = nextGameState.getPacmanPosition()
    oldFood = currGameState.getNumFood()
    newFood = nextGameState.getNumFood()
    remainingFood = nextGameState.getFood()
    newGhostStates = nextGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    ghosts = nextGameState.getGhostPositions()

    scaredTimesInSecond = newGhostStates[0].scaredTimer

    "*** YOUR CODE HERE ***"
    score = 0
    if (newFood < oldFood):
      score += 1
    if oldPos == newPos:
      score -= 15

    for ghost in newGhostStates:
      distanceFromGhost = util.manhattanDistance(newPos, ghost.getPosition())
      # Si on est proche d'un phantome pendant qu'ils sont mangeable c'est vraiment bien
      if distanceFromGhost <= 3 and ghost.scaredTimer >= 2:
        score += 200
      # Si on est proche et qu'ils sont pas mangeable c'est moins top d'un coups
      elif distanceFromGhost <= 1:
            score -= 20
    return nextGameState.getScore() +  score
