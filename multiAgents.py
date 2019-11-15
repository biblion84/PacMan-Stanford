# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance, Queue
from game import Directions, Actions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currGameState, pacManAction):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
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
    #Si il mange c'est mieux, logique mais redondant car on prend le getScore au final
    if (newFood < oldFood):
      score += 1

    #Il faut se rapprocher de la bouffe
    # nearestFoodDistanceNew = float('inf')
    # for food in remainingFood:
    #   newDistance = util.manhattanDistance(food, newPos)
    #   if newDistance < nearestFoodDistanceNew:
    #     nearestFoodDistanceNew = newDistance
    #
    # nearestFoodDistanceOld =  float('inf')
    # for food in remainingFood:
    #   newDistance = util.manhattanDistance(food, oldPos)
    #   if newDistance < nearestFoodDistanceOld:
    #     nearestFoodDistanceOld = newDistance
    #
    # if nearestFoodDistanceNew < nearestFoodDistanceOld:
    #   score += 10

    #The closer you are to food the better it is

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

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, currGameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      currGameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      currGameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      currGameState.getNumAgents():
        Returns the total number of agents in the game
    """

    # Get all possible move by pacman
    # Score them

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
        #value = float('inf')
        actions = gameState.getLegalActions(agent)
        values = []
        for action in actions:
          values.append(expectiMax(gameState.generateSuccessor(agent, action), depth, alpha, beta,nextAgent))
        return (sum(values)/len(values))
        beta = min(beta, value)
        return value

    legalActions = currGameState.getLegalActions(0)
    legalActions.remove(Directions.STOP)
    bestAction = legalActions[0]
    valueOfAction = -float('inf')
    for action in legalActions:
      value = expectiMax(currGameState.generateSuccessor(0, action), self.depth, -float('inf'),float('inf'), 1)
      if value > valueOfAction: # Here we can see that value == value of action a lot
        bestAction = action # I need to change the evaluation function to change that
        valueOfAction = value
    HistoActions.append(bestAction)
    return  bestAction

def betterEvaluationFunction(currGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  pacman = currGameState.getPacmanState()
  position = pacman.getPosition()
  foods = currGameState.getFood()
  foodRemaining = currGameState.getNumFood()
  walls = currGameState.getWalls()
  capsules = currGameState.getCapsules()
  ghosts = currGameState.getGhostStates()
  capsulesGrid = currGameState.getFood()

  for x in range(0,capsulesGrid.width -1):
    for y in range(0, capsulesGrid.height - 1):
      capsulesGrid[x][y] = False

  for capsule in capsules:
    capsulesGrid[capsule[0]][capsule[1]] = True

  pacmanConfDir = pacman.configuration.direction
  badposition = BadAction(HistoActions)

  scoreBadPos = 0
  if pacmanConfDir == badposition:
      scoreBadPos = -10
  # for capsule in capsules:
  #   foods[capsule[0]][capsule[1]] = True

  # manhattanFood = nearestFood(foods, position)
  # nearestFoodToPacman = manhattanFood
  nearestFoodToPacman = nearestFoodGansterDjikstra(position, walls, foods)
  if nearestFoodToPacman ==  200:
    nearestFoodToPacman = nearestFood(foods, position)

  if (foodRemaining < 1) :
      nearestFoodToPacman = nearestFood(foods, position)
  # foodScore = 1000
  # if (foodRemaining > 0):
  #   foodScore = float(1/foodRemaining)
  nGhost = nearestGhost(ghosts, position)
  fleeingScore = 0
  if util.manhattanDistance(nGhost.getPosition(), position) <= 1:
    if (nGhost.scaredTimer > 2):
      fleeingScore += 100
    else :
      fleeingScore = -10

  scaredGhost = 0
  # for ghost in ghosts:
  #   if ghost.scaredTimer > 3:
  #     distanceFromGhost = util.manhattanDistance(position, ghost.getPosition())
  #     scoreFantome += 1/distanceFromGhost * -2


  capsulesNumber = len(capsules)
  # capsuleNear = 0
  # for capsule in capsules:
  #     capsuleNear = max(capsuleNear, 1/util.manhattanDistance(position, [capsule[0], capsule[1]]))

  # if (capsulesNumber > 0 )
  return  currGameState.getScore() + (float(1)/(nearestFoodToPacman) * 5) + fleeingScore + capsulesNumber * -20 + scoreBadPos

def nearestGhost(ghosts, position):
  distance = float('inf')
  nGhost = ghosts[0]
  for ghost in ghosts:
    newDistance = util.manhattanDistance(position, ghost.getPosition())
    if newDistance < distance:
      distance = newDistance
      nGhost = ghost
  return nGhost

def nearestFoodGansterDjikstra(pacmanPosition, walls, foods):
    depth = 1
    possiblePositions = Actions.getLegalNeighbors(pacmanPosition, walls)
    positionQueue = Queue()
    for position in possiblePositions:
        positionQueue.push(position)
    found = False
    positionTested = []
    while not found and not positionQueue.isEmpty():
        positionToTest = positionQueue.pop()
        if foods[positionToTest[0]][positionToTest[1]] or depth > 200:
            found = True
        else :
            possiblePositions = Actions.getLegalNeighbors(positionToTest, walls)
            possiblePositions.remove(positionToTest)
            positionTested.append(positionToTest)
            for position in possiblePositions:
                if not (position in positionTested):
                    positionQueue.push(position)
            depth += 1
    return depth
        #
        # for position in possiblePositions:
        #     if foods[position.x][position.y]:
        #         return depth
        # for position in possiblePositions:
        #         return nearestFoodGansterDjikstra(position, walls, foods, depth)


def nearestFood(foods, position):
  distance = float('inf')
  for x in range(0, foods.width):
    for y in range(0, foods.height):
      if foods[x][y]:
        distance = min(distance, util.manhattanDistance(position, [x, y]))

  # for food in foods:
  #   distance = min(distance, util.manhattanDistance(food.getPosition(), position))
  return distance

def BadAction(LastAction):
    taile = len(LastAction)
    taile -= 1
    modif = False
    lastposition = 'Stop'
    if taile >= 0:
      lastposition = LastAction[taile]
    if lastposition == 'West' and modif == False:
      lastposition = 'East'
      modif = True
    if lastposition == 'East' and modif == False:
      lastposition = 'West'
      modif = True
    if lastposition == 'North' and modif == False:
      lastposition = 'South'
      modif = True
    if lastposition == 'South' and modif == False:
      lastposition = 'North'
      modif = True
    return lastposition
HistoActions = []

# Abbreviation
better = betterEvaluationFunction

