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
from collections import defaultdict
import math
from game import Agent
from MonteCarlo import MCTS, Node
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.linear_model import Ridge

dataColumns = ["ghostUp", "ghostDown", "ghostLeft", "ghostRight", "wallUp", "wallDown", "wallLeft", "wallRight",
               "foodUp", "foodDown", "foodLeft", "foodRight", "emptyUp", "emptyDown", "emptyLeft", "emptyRight",
               "nearestFood", "nearestGhost", "nearestCapsule", "legalPositionUp", "legalPositionDown",
               "legalPositionULeft", "legalPositionRight", "pacmanPositionX", "pacmanPositionY", "labelNextAction"]

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
    self.indexDataframe = 0
    self.alreadyWroteHeaders = False
    self.dataFrame = pd.DataFrame(columns=["ghostUp","ghostDown","ghostLeft","ghostRight","wallUp","wallDown","wallLeft","wallRight","foodUp","foodDown","foodLeft","foodRight","emptyUp","emptyDown","emptyLeft","emptyRight","nearestFood","nearestGhost","nearestCapsule","legalPositionUp","legalPositionDown","legalPositionULeft","legalPositionRight","pacmanPositionX","pacmanPositionY","lastAction" ,"labelNextAction"])


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
    nextGameState = False
    for action in legalActions:
      nextGS = currGameState.generateSuccessor(0, action)
      value = expectiMax(nextGS, self.depth, -float('inf'),float('inf'), 1)
      value -=  AlreadyVisitedScore(nextGS.getPacmanPosition()) * 0.1
      if value > valueOfAction: # Here we can see that value == value of action a lot
        bestAction = action # I need to change the evaluation function to change that
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
          else :
            self.dataFrame.to_csv(f, header=True)
            self.alreadyWroteHeaders = True
      else:
        self.dataFrame = pd.DataFrame(
          columns=["ghostUp", "ghostDown", "ghostLeft", "ghostRight", "wallUp", "wallDown", "wallLeft", "wallRight",
                   "foodUp", "foodDown", "foodLeft", "foodRight", "emptyUp", "emptyDown", "emptyLeft", "emptyRight",
                   "nearestFood", "nearestGhost", "nearestCapsule", "legalPositionUp", "legalPositionDown",
                   "legalPositionULeft", "legalPositionRight", "pacmanPositionX", "pacmanPositionY", "lastAction", "labelNextAction"])
    return  bestAction

def AlreadyVisitedScore(position):
  countAlreadyVisited = AlreadyVisited.count(position)
  if countAlreadyVisited == 0:
    return 0
  else :
    return 1 - (float(1) / countAlreadyVisited)

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
  ghostsGrid = foods.copy()
  capsulessGrid = foods.copy()

  for ghostRow in ghostsGrid:
    for ghostCol in range(0, len(ghostRow) -1):
      ghostRow[ghostCol] = False

  for ghost in ghosts:
    ghostPosition = ghost.getPosition()
    x = int(ghostPosition[0])
    y = int(ghostPosition[1])
    ghostsGrid[x][y] = True

  for capsuleRow in capsulessGrid:
    for capsuleCol in range(0, len(capsuleRow) -1):
      capsuleRow[capsuleCol] = False

  for capsule in capsules:
    capsulessGrid[capsule[0]][capsule[1]] = True



  scoreGhostAfraid = 0
  # for ghost in ghosts:
  #   if ghost.scaredTimer > 2:
  #     scoreGhostAfraid += 50

  pacmanConfDir = pacman.configuration.direction
  badposition = BadAction(HistoActions)

  scoreBadPos = 0
  # if pacmanConfDir == badposition:
  #     scoreBadPos = -5

  # if position in AlreadyVisited:
  #   scoreBadPos -= 1

  nearestFoodToPacman = nearestFoodGansterDjikstra(position, walls, foods) + nearestFood(foods, position)
  if nearestFoodToPacman ==  200:
    nearestFoodToPacman = nearestFood(foods, position)

  if (foodRemaining < 1) :
      nearestFoodToPacman = nearestFood(foods, position)

  nGhost = nearestGhost(ghosts, position, walls, ghostsGrid.copy())
  fleeingScore = 0
  ghostScared = False
  scaredGhosts = []
  for ghost in ghosts:
    # if ghost.scaredTimer > util.manhattanDistance(position, ghost.getPosition()):
      # scaredGhosts.append(ghost)
    if ghost.scaredTimer > 0 :
      ghostScared = True
      scaredGhosts.append(ghost)

  distanceNearestGhost = nearestFoodGansterDjikstra(position, walls, ghostsGrid)

  if distanceNearestGhost <= 2:
    if (nGhost.scaredTimer > 2):
      fleeingScore += 100
    else :
      fleeingScore = -10

  # maxDistanceFromGhost = util.manhattanDistance(farthestGhost(ghosts, position).getPosition(), position)
  capsuleScore = 0
  capsulesNumber = len(capsules)
  if (capsulesNumber > 0 and not ghostScared):
    nearestCapsule = nearestFoodGansterDjikstra(position, walls, capsulessGrid)
    nearestFoodToPacman = nearestCapsule
  else :
    capsuleScore = 15

  maybeTrapped = 0
  if len(currGameState.getLegalActions(0)) == 2:
    maybeTrapped = -10

  if len(scaredGhosts) > 0:
    nearestFoodToPacman = distanceNearestGhost
    # nearestFoodToPacman = util.manhattanDistance(nearestGhost(scaredGhosts, position).getPosition(), position) / 10

  return  currGameState.getScore() +\
          (float(1)/(nearestFoodToPacman) * 5) +\
          fleeingScore +\
          scoreGhostAfraid +\
          capsuleScore +\
          maybeTrapped +\
          scoreBadPos

def nearestGhost(ghosts, position, walls, ghostsGrid):
  for ghostRow in ghostsGrid:
    for ghostCol in range(0, len(ghostRow) -1):
      ghostRow[ghostCol] = False

  for ghost in ghosts:
    ghostPosition = ghost.getPosition()

  distance = float('inf')
  nGhost = ghosts[0]
  for ghost in ghosts:
    x = int(ghostPosition[0])
    y = int(ghostPosition[1])
    ghostsGrid[x][y] = True
    newDistance = nearestFoodGansterDjikstra(position, walls, ghostsGrid)
    if newDistance < distance:
      distance = newDistance
      nGhost = ghost
    ghostsGrid[x][y] = False
  return nGhost


def farthestGhost(ghosts, position):
  distance = float('-inf')
  nGhost = ghosts[0]
  for ghost in ghosts:
    newDistance = util.manhattanDistance(position, ghost.getPosition())
    if newDistance > distance:
      distance = newDistance
      nGhost = ghost
  return nGhost



def nearestFoodGansterDjikstra(pacmanPosition, walls, foods):
    depth = 1
    positionTested = []
    positionTested.append(pacmanPosition)
    possiblePositions = Actions.getLegalNeighbors(pacmanPosition, walls)

    for positionAlreadyTested in positionTested:
      if (positionAlreadyTested in possiblePositions):
        possiblePositions.remove(positionAlreadyTested)

    positionQueue = Queue()
    positionQueueA = Queue()
    positionQueueB = Queue()
    for position in possiblePositions:
        positionQueueA.push(position)

    found = False

    positionQueue = positionQueueA
    QueueA = True

    while not found:
        if positionQueue.isEmpty():
          return depth
        positionToTest = positionQueue.pop()
        if (positionQueue.isEmpty()):
          depth += 1
          if (QueueA):
            QueueA = False
            positionQueue = positionQueueB
          else:
            QueueA = True
            positionQueue = positionQueueA
        positionTested.append(positionToTest)
        if foods[positionToTest[0]][positionToTest[1]] or depth > 200:
            found = True
        else :
            possiblePositions = Actions.getLegalNeighbors(positionToTest, walls)
            possiblePositions.remove(positionToTest)
            for position in possiblePositions:
                if not (position in positionTested):
                  if (QueueA):
                    positionQueueB.push(position)
                  else:
                    positionQueueA.push(position)

    return depth


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


AlreadyVisited = []
HistoActions = []

# Abbreviation
better = betterEvaluationFunction



## MONTE CARLO
def ghostHere(ghosts, position):
  for ghost in ghosts:
    if ghost.getPosition() == position:
      return True
  return False

def wallHere(walls, position):
  return walls[position[0]][position[1]]

def foodHere(foods, position):
  return foods[position[0]][position[1]]

def getActionsNumber(action):
  if action == "North":
    return 0
  if action == "South":
    return 1
  if action == "West":
    return 2
  if action == "East":
    return 3

def getActionByNumber(number):
  if number == 0:
    return "North"
  if number == 1:
    return "South"
  if number == 2:
    return"West"
  if number == 3:
    return"East"

def extractFeature(gameState, actionChoosed):
  # ce que y'a dans case haut,bas,gauche,droite
  # distance entre bouffe plus proche
  # distance entre pacman et fantome
  pacman = gameState.getPacmanState()
  pacmanPosition = gameState.getPacmanPosition()
  ghosts = gameState.getGhostStates()
  walls = gameState.getWalls()
  foods = gameState.getFood()
  foodNumber = gameState.getNumFood()
  capsules = gameState.getCapsules()
  ghostsGrid = foods.copy()
  capsulesGrid = foods.copy()

  for ghostRow in ghostsGrid:
    for ghostCol in range(0, len(ghostRow) -1):
      ghostRow[ghostCol] = False

  for ghost in ghosts:
    ghostPosition = ghost.getPosition()
    x = int(ghostPosition[0])
    y = int(ghostPosition[1])
    ghostsGrid[x][y] = True
    
  for capsuleRow in capsulesGrid:
    for capsuleCol in range(0, len(capsuleRow) -1):
      capsuleRow[capsuleCol] = False

  for capsule in capsules:
    capsulesGrid[capsule[0]][capsule[1]] = True
  
  ghostUp = ghostHere(ghosts, [pacmanPosition[0], pacmanPosition[1] + 1])
  ghostDown = ghostHere(ghosts, [pacmanPosition[0], pacmanPosition[1] - 1])
  ghostLeft = ghostHere(ghosts, [pacmanPosition[0] - 1, pacmanPosition[1]])
  ghostRight = ghostHere(ghosts, [pacmanPosition[0] + 1, pacmanPosition[1]])

  wallUp = wallHere(walls, [pacmanPosition[0], pacmanPosition[1] + 1])
  wallDown = wallHere(walls, [pacmanPosition[0], pacmanPosition[1] - 1])
  wallLeft = wallHere(walls, [pacmanPosition[0] - 1, pacmanPosition[1]])
  wallRight = wallHere(walls, [pacmanPosition[0] + 1, pacmanPosition[1]])
  
  foodUp = foodHere(foods, [pacmanPosition[0], pacmanPosition[1] + 1])
  foodDown = foodHere(foods, [pacmanPosition[0], pacmanPosition[1] - 1])
  foodLeft = foodHere(foods, [pacmanPosition[0] - 1, pacmanPosition[1]])
  foodRight = foodHere(foods, [pacmanPosition[0] + 1, pacmanPosition[1]])
  
  emptyUp = 1 if  (ghostUp + wallUp + foodUp) >= 0 else 0
  emptyDown = 1 if  (ghostDown + wallDown + foodDown) >= 0 else 0
  emptyLeft = 1 if  (ghostLeft + wallLeft + foodLeft) >= 0 else 0
  emptyRight = 1 if  (ghostRight + wallRight + foodRight) >= 0 else 0
  #nearestGhost(ghosts, position, walls, ghostsGrid):
  nearestFood =  float(1) / nearestFoodGansterDjikstra(pacmanPosition, walls, foods) if foodNumber > 0  else 0
  #nearestGhost = 1 / nearestGhost(ghosts, pacmanPosition, walls, ghostsGrid)
  nearestGhostDistance = float(1) / nearestFoodGansterDjikstra(pacmanPosition, walls, ghostsGrid)
  nearestCapsule = float(1) / nearestFoodGansterDjikstra(pacmanPosition, walls, capsulesGrid) if (len(capsules) > 0) else 0
  nextAction = getActionsNumber(actionChoosed)
  lastAction = getActionsNumber(BadAction(HistoActions))

  # nextActionUp = "North" == nextAction
  # nextActionDown = "South" == nextAction
  # nextActionLeft = "West" == nextAction
  # nextActionRight = "East" == nextAction
  
  legalPositions = gameState.getLegalActions(0)
  
  legalPositionUp = "North" in legalPositions
  legalPositionDown = "South" in legalPositions
  legalPositionULeft = "West" in legalPositions
  legalPositionRight = "East" in legalPositions
  
  pacmanPositionX = float(pacmanPosition[0])  / gameState.data.layout.width
  pacmanPositionY = float(pacmanPosition[1]) /  gameState.data.layout.height
  
  dataFrameCurrentState = [ ghostUp ,
                              ghostDown ,
                              ghostLeft ,
                              ghostRight ,
                              wallUp ,
                              wallDown ,
                              wallLeft ,
                              wallRight ,
                              foodUp ,
                              foodDown ,
                              foodLeft ,
                              foodRight ,
                              emptyUp ,
                              emptyDown ,
                              emptyLeft ,
                              emptyRight ,
                              nearestFood ,
                              nearestGhostDistance ,
                              nearestCapsule ,
                              legalPositionUp ,
                              legalPositionDown ,
                              legalPositionULeft ,
                              legalPositionRight ,
                              pacmanPositionX ,
                              pacmanPositionY,
                              lastAction,
                                nextAction]
  return dataFrameCurrentState
#TODO Nombre de bouffe totale en haut, gauche, droite

class MonteCarloAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """
  tree = MCTS()
  gameState = None

  def getAction(self, currGameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    if self.gameState is None:
      self.gameState = Node(currGameState)
      for _ in range(3):
        self.tree.do_rollout(self.gameState)

    # for _ in range(1):
    #   self.tree.do_rollout(Node(currGameState))

    nextMoveBoardState = self.tree.choose(Node(currGameState))
    return nextMoveBoardState.gameState.data.agentStates[0].configuration.direction

