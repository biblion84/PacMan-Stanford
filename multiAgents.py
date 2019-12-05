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
from MonteCarlo import MCTS, Node
import pandas as pd
import numpy as np
from joblib import load


dataColumns =  ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23",
               "nearestFood", "nearestGhost", "nearestCapsule", "nearestGhostAfraid", "lastAction", "labelNextAction"]

dataColumnsDistanceOnly =  ["foodUp","foodDown","foodLeft","foodRight","ghostUp","ghostDown","ghostLeft","ghostRight",
                            "wallUp","wallDown","wallLeft","wallRight","lastAction", "labelNextAction"]
def scoreEvaluationFunction(currentGameState):
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2', filesave = "Reflex0.csv"):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)
    self.indexDataframe = 0
    self.alreadyWroteHeaders = False
    self.filesave = filesave
    self.dataFrame = pd.DataFrame(columns=dataColumns)


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

def nearestGhostDumb(ghosts, position):
  distance = float('-inf')
  nGhost = ghosts[0]
  for ghost in ghosts:
    newDistance = util.manhattanDistance(position, ghost.getPosition())
    if newDistance < distance:
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

def ghostHere(ghosts, position):
  for ghost in ghosts:
    if ghost.getPosition() == position:
      return True
  return False

def inGrid(foods, position):
  if foods is None:
    return False
  if (position[0] >= len(foods.data)):
    return False
  if (position[1] >= len(foods[00])):
    return False
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
  if action == "Stop":
    return 4
  return 0

def getActionByNumber(number):
  if number == 0:
    return "North"
  if number == 1:
    return "South"
  if number == 2:
    return"West"
  if number == 3:
    return"East"
  if number == 4:
    return "Stop"

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

  wallUp = inGrid(walls, [pacmanPosition[0], pacmanPosition[1] + 1])
  wallDown = inGrid(walls, [pacmanPosition[0], pacmanPosition[1] - 1])
  wallLeft = inGrid(walls, [pacmanPosition[0] - 1, pacmanPosition[1]])
  wallRight = inGrid(walls, [pacmanPosition[0] + 1, pacmanPosition[1]])
  
  foodUp = inGrid(foods, [pacmanPosition[0], pacmanPosition[1] + 1])
  foodDown = inGrid(foods, [pacmanPosition[0], pacmanPosition[1] - 1])
  foodLeft = inGrid(foods, [pacmanPosition[0] - 1, pacmanPosition[1]])
  foodRight = inGrid(foods, [pacmanPosition[0] + 1, pacmanPosition[1]])

  emptyUp = 1 if  (ghostUp + wallUp + foodUp) >= 0 else 0
  emptyDown = 1 if  (ghostDown + wallDown + foodDown) >= 0 else 0
  emptyLeft = 1 if  (ghostLeft + wallLeft + foodLeft) >= 0 else 0
  emptyRight = 1 if  (ghostRight + wallRight + foodRight) >= 0 else 0
  #nearestGhost(ghosts, position, walls, ghostsGrid):
  nearestFood =  float(1) / nearestFoodGansterDjikstra(pacmanPosition, walls, foods) if foodNumber > 0  else 0
  nGhost = nearestGhost(ghosts, pacmanPosition, walls, ghostsGrid)
  nearestGhostDistance = float(1) / nearestFoodGansterDjikstra(pacmanPosition, walls, ghostsGrid)
  nearestCapsule = float(1) / nearestFoodGansterDjikstra(pacmanPosition, walls, capsulesGrid) if (len(capsules) > 0) else 0
  nextAction = getActionsNumber(actionChoosed)
  #produit scalaire du rapport de la direction de pacman avec le fantome pour savoir si il vont se catapulter

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
  
  dataFrameCurrentState = getSurroundingMatrix(gameState) + \
                          [
                              nearestFood ,
                              nearestGhostDistance ,
                              nearestCapsule ,
                              nGhost.scaredTimer > 0,
                              lastAction[0],
                                nextAction]
  return dataFrameCurrentState

def distanceWithAction(gs, action):
  if action not in gs.getLegalActions(0):
    return   [-100,-100]
  gameState = gs.generatePacmanSuccessor(action)
  pacmanPosition = gameState.getPacmanPosition()
  ghosts = gameState.getGhostStates()
  walls = gameState.getWalls()
  foods = gameState.getFood()
  foodNumber = gameState.getNumFood()
  capsules = gameState.getCapsules()
  ghostsGrid = foods.copy()
  capsulesGrid = foods.copy()

  for ghostRow in ghostsGrid:
    for ghostCol in range(0, len(ghostRow) - 1):
      ghostRow[ghostCol] = False

  for ghost in ghosts:
    ghostPosition = ghost.getPosition()
    x = int(ghostPosition[0])
    y = int(ghostPosition[1])
    ghostsGrid[x][y] = True
  nearestFood = nearestFoodGansterDjikstra(pacmanPosition, walls, foods) if foodNumber > 0 else 0
  nGhost = nearestGhostDumb(ghosts, pacmanPosition)
  # nearestGhostDistance = nearestFoodGansterDjikstra(pacmanPosition, walls, ghostsGrid)
  nearestGhostDistance = util.manhattanDistance(nGhost.getPosition(), pacmanPosition)

  return   [nearestFood, nearestGhostDistance]

def extractFeatureDistanceOnly(gameState, actionChoosed):
  ghosts = gameState.getGhostStates()
  foods = gameState.getFood()
  ghostsGrid = foods.copy()

  pacmanPosition = gameState.getPacmanPosition()
  walls = gameState.getWalls()

  for ghostRow in ghostsGrid:
    for ghostCol in range(0, len(ghostRow) - 1):
      ghostRow[ghostCol] = False

  for ghost in ghosts:
    ghostPosition = ghost.getPosition()
    x = int(ghostPosition[0])
    y = int(ghostPosition[1])
    ghostsGrid[x][y] = True
    
  wallUp = inGrid(walls, [pacmanPosition[0], pacmanPosition[1] + 1]) or inGrid(ghostsGrid, [pacmanPosition[0], pacmanPosition[1] + 1])
  wallDown = inGrid(walls, [pacmanPosition[0], pacmanPosition[1] - 1]) or inGrid(ghostsGrid, [pacmanPosition[0], pacmanPosition[1] - 1])
  wallLeft = inGrid(walls, [pacmanPosition[0] - 1, pacmanPosition[1]]) or inGrid(ghostsGrid, [pacmanPosition[0] - 1, pacmanPosition[1]])
  wallRight = inGrid(walls, [pacmanPosition[0] + 1, pacmanPosition[1]]) or inGrid(ghostsGrid, [pacmanPosition[0] + 1, pacmanPosition[1]])

  nextAction = getActionsNumber(actionChoosed)
  # produit scalaire du rapport de la direction de pacman avec le fantome pour savoir si il vont se catapulter

  distancesUp = distanceWithAction(gameState, "North")
  distancesDown = distanceWithAction(gameState, "South")
  distancesLeft = distanceWithAction(gameState, "West")
  distancesRight = distanceWithAction(gameState, "East")

  dataFrameCurrentState =  [
                            distancesUp[0],
                            distancesDown[0],
                            distancesLeft[0],
                            distancesRight[0],
                            distancesUp[1],
                            distancesDown[1],
                            distancesLeft[1],
                            distancesRight[1],
                            wallUp,
                            wallDown,
                            wallLeft,
                            wallRight,
                            lastAction[0],
                            nextAction]
  return dataFrameCurrentState
#TODO Nombre de bouffe totale en haut, gauche, droite

def AlreadyVisitedScore(position):
  countAlreadyVisited = AlreadyVisited.count(position)
  if countAlreadyVisited == 0:
    return 0
  else:
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
    for ghostCol in range(0, len(ghostRow) - 1):
      ghostRow[ghostCol] = False
  
  for ghost in ghosts:
    ghostPosition = ghost.getPosition()
    x = int(ghostPosition[0])
    y = int(ghostPosition[1])
    ghostsGrid[x][y] = True
  
  for capsuleRow in capsulessGrid:
    for capsuleCol in range(0, len(capsuleRow) - 1):
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
  if nearestFoodToPacman == 200:
    nearestFoodToPacman = nearestFood(foods, position)
  
  if (foodRemaining < 1):
    nearestFoodToPacman = nearestFood(foods, position)
  
  nGhost = nearestGhost(ghosts, position, walls, ghostsGrid.copy())
  fleeingScore = 0
  ghostScared = False
  scaredGhosts = []
  for ghost in ghosts:
    # if ghost.scaredTimer > util.manhattanDistance(position, ghost.getPosition()):
    # scaredGhosts.append(ghost)
    if ghost.scaredTimer > 0:
      ghostScared = True
      scaredGhosts.append(ghost)
  
  distanceNearestGhost = nearestFoodGansterDjikstra(position, walls, ghostsGrid)
  
  if distanceNearestGhost <= 2:
    if (nGhost.scaredTimer > 2):
      fleeingScore += 100
    else:
      fleeingScore = -10
  
  # maxDistanceFromGhost = util.manhattanDistance(farthestGhost(ghosts, position).getPosition(), position)
  capsuleScore = 0
  capsulesNumber = len(capsules)
  if (capsulesNumber > 0 and not ghostScared):
    nearestCapsule = nearestFoodGansterDjikstra(position, walls, capsulessGrid)
    nearestFoodToPacman = nearestCapsule
  else:
    capsuleScore = 15
  
  maybeTrapped = 0
  if len(currGameState.getLegalActions(0)) <= 2:
    maybeTrapped = -10
  
  if len(scaredGhosts) > 0:
    nearestFoodToPacman = distanceNearestGhost
    # nearestFoodToPacman = util.manhattanDistance(nearestGhost(scaredGhosts, position).getPosition(), position) / 10
  
  return currGameState.getScore() + \
         (float(1) / (nearestFoodToPacman) * 5) + \
         fleeingScore + \
         scoreGhostAfraid + \
         capsuleScore + \
         maybeTrapped + \
         scoreBadPos


def betterEvaluationFunctionReflex(currGameState):
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
    for ghostCol in range(0, len(ghostRow) - 1):
      ghostRow[ghostCol] = False

  for ghost in ghosts:
    ghostPosition = ghost.getPosition()
    x = int(ghostPosition[0])
    y = int(ghostPosition[1])
    ghostsGrid[x][y] = True

  for capsuleRow in capsulessGrid:
    for capsuleCol in range(0, len(capsuleRow) - 1):
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
  if nearestFoodToPacman == 200:
    nearestFoodToPacman = nearestFood(foods, position)

  # if (foodRemaining < 1):
  #   nearestFoodToPacman = nearestFood(foods, position)

  nGhost = nearestGhost(ghosts, position, walls, ghostsGrid.copy())
  fleeingScore = 0
  ghostScared = False
  scaredGhosts = []
  for ghost in ghosts:
    # if ghost.scaredTimer > util.manhattanDistance(position, ghost.getPosition()):
    # scaredGhosts.append(ghost)
    if ghost.scaredTimer > 0:
      ghostScared = True
      scaredGhosts.append(ghost)

  distanceNearestGhost = nearestFoodGansterDjikstra(position, walls, ghostsGrid)

  if distanceNearestGhost <= 2:
    if (nGhost.scaredTimer > 2):
      fleeingScore += 100
    else:
      fleeingScore = -10

  if distanceNearestGhost <= 1:
    if (nGhost.scaredTimer > 1):
      fleeingScore += 100
    else:
      fleeingScore = -10

  # maxDistanceFromGhost = util.manhattanDistance(farthestGhost(ghosts, position).getPosition(), position)
  capsuleScore = 0
  # capsulesNumber = len(capsules)
  # if (capsulesNumber > 0 and not ghostScared):
  #   nearestCapsule = nearestFoodGansterDjikstra(position, walls, capsulessGrid)
  #   if (nearestCapsule > 1):
  #     nearestFoodToPacman = nearestCapsule
  # else:
  #   capsuleScore = 0.1

  maybeTrapped = 0
  if len(currGameState.getLegalActions(0)) <= 2:
    maybeTrapped = -7

  # if len(scaredGhosts) > 0:
  #   nearestFoodToPacman = distanceNearestGhost
    # nearestFoodToPacman = util.manhattanDistance(nearestGhost(scaredGhosts, position).getPosition(), position) / 10

  return currGameState.getScore() + \
         (float(1) / (nearestFoodToPacman) * 5) - \
         fleeingScore + \
         scoreGhostAfraid + \
         capsuleScore + \
         maybeTrapped + \
         scoreBadPos


lastAction = [0]
AlreadyVisited = []
HistoActions = []


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

    oldSore = currGameState.getScore()
    newScore = currGameState.generateSuccessor(0, bestAction).getScore()
    if (newScore > oldSore):
        features = extractFeature(currGameState, bestAction)
        self.dataFrame.loc[self.indexDataframe, :] = features
        lastAction[0] = getActionsNumber(bestAction)
        self.indexDataframe = self.indexDataframe + 1
    
    if (nextGameState.isWin()):
      if (nextGameState.getScore() > 1500):
        with open('dataGameWonMoreThan1500WithColumnNames.csv', 'a') as f:
          # self.dataFrame.columns = ["ghostUp","ghostDown","ghostLeft","ghostRight","wallUp","wallDown","wallLeft","wallRight","foodUp","foodDown","foodLeft","foodRight","emptyUp","emptyDown","emptyLeft","emptyRight","nearestFood","nearestGhost","nearestCapsule","legalPositionUp","legalPositionDown","legalPositionULeft","legalPositionRight","pacmanPositionX","pacmanPositionY","labelNextAction"]
          
          if (self.alreadyWroteHeaders):
            self.dataFrame.to_csv(f, header=False, index=False)
          else:
            self.dataFrame.to_csv(f, header=True, index=False)
            self.alreadyWroteHeaders = True
    return bestAction

def getMatrixDataframe(matrix, actionTaken):
  if not actionTaken >= 0:
    actionTaken = 0
  matrix.append(getActionsNumber(actionTaken))
  return  matrix


class ReflexAgent(Agent):
  
  
  def getAction(self, gameState):
    pacmanPosition = gameState.getPacmanPosition()
    ghosts = gameState.getGhostStates()
    walls = gameState.getWalls()
    foods = gameState.getFood()
    foodNumber = gameState.getNumFood()
    capsules = gameState.getCapsules()
    ghostsGrid = foods.copy()
    capsulesGrid = foods.copy()
  
    for ghostRow in ghostsGrid:
      for ghostCol in range(0, len(ghostRow) - 1):
        ghostRow[ghostCol] = False
  
    for ghost in ghosts:
      ghostPosition = ghost.getPosition()
      x = int(ghostPosition[0])
      y = int(ghostPosition[1])
      ghostsGrid[x][y] = True
      
    legalMoves = gameState.getLegalActions()
    # legalMoves.remove(Directions.STOP)
    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    bestAction = legalMoves[chosenIndex]
    if bestAction == None:
      bestAction = legalMoves[0]
    # matrix = getSurroundingMatrix(gameState)
    # matrixDataframe = getMatrixDataframe(matrix, bestAction)
    nextGameState = gameState.generatePacmanSuccessor(bestAction)
    AlreadyVisited.append(nextGameState.getPacmanPosition())
    # if self.evaluationFunction(gameState, bestAction) > gameState.getScore() or   nearestFoodGansterDjikstra(pacmanPosition, walls, ghostsGrid) == 1:
      # self.dataFrameMatrix.loc[self.indexDataframe, :] =  matrixDataframe
    if (not nextGameState.isLose()):
      self.dataFrameDistance.loc[self.indexDataframe, :] = extractFeatureDistanceOnly(gameState, bestAction)
      lastAction[0] = getActionsNumber(bestAction)
      self.indexDataframe = self.indexDataframe + 1
    
    if (nextGameState.isWin() or nextGameState.isLose()):
      if True:
        with open(self.filesave, 'a') as f:
          # self.dataFrame.columns = ["ghostUp","ghostDown","ghostLeft","ghostRight","wallUp","wallDown","wallLeft","wallRight","foodUp","foodDown","foodLeft","foodRight","emptyUp","emptyDown","emptyLeft","emptyRight","nearestFood","nearestGhost","nearestCapsule","legalPositionUp","legalPositionDown","legalPositionULeft","legalPositionRight","pacmanPositionX","pacmanPositionY","labelNextAction"]
      
          if (self.alreadyWroteHeaders):
            self.dataFrameDistance.to_csv(f, header=False, index=False)
          else:
            self.dataFrameDistance.to_csv(f, header=True, index=False)
            self.alreadyWroteHeaders = True
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
    score -= AlreadyVisitedScore(nextGameState.getPacmanPosition()) * 0.5
    for ghost in newGhostStates:
      distanceFromGhost = util.manhattanDistance(newPos, ghost.getPosition())
      # # Si on est proche d'un phantome pendant qu'ils sont mangeable c'est vraiment bien
      # if distanceFromGhost <= 3 and ghost.scaredTimer >= 2:
      #   score += 200
      # Si on est proche et qu'ils sont pas mangeable c'est moins top d'un coups
      if distanceFromGhost <= 1:
            score -= 40
    return  betterEvaluationFunctionReflex(currGameState.generatePacmanSuccessor(pacManAction)) +  score



def outOfMapX(x, layout):
  if x >= 0 and x <= layout.width:
    return True
  else:
    return False
  
def outOfMap(x, y, layout):
  if y >= 0 and y <= layout.height and x >= 0 and x <= layout.width:
    return False
  else:
    return True

def getCapsuleGrid(foods, capsules):
  capsulesGrid = foods.copy()
  
  for capsuleRow in capsulesGrid:
    for capsuleCol in range(0, len(capsuleRow) - 1):
      capsuleRow[capsuleCol] = False
  
  for capsule in capsules:
    capsulesGrid[capsule[0]][capsule[1]] = True
    
    return  capsulesGrid

def getSurroundingMatrix(gameState):
  ghosts = gameState.getGhostStates()
  walls = gameState.getWalls()
  foods = gameState.getFood()
  capsules = getCapsuleGrid(foods, gameState.getCapsules())
  
  initialPacmanPosition = gameState.getPacmanState().getPosition()
  xP = initialPacmanPosition[0]
  yP = initialPacmanPosition[1]
  matrix = []
  for x in range(-2, 3):
    for y in range(-2,3):
      xTest = xP + x
      yTest = yP + y
      
      if outOfMap(xTest, yTest, gameState.data.layout):
        matrix.append(0)
        continue
      if xTest == xP and yTest == yP:
        continue
        
      bitmap = 2 # 2^1
      if inGrid(foods, [xTest, yTest]):
        bitmap += 4 # 2^2
      if inGrid(capsules, [xTest, yTest]):
        bitmap += 8 # 2^3
      if ghostHere(ghosts, [xTest, yTest]):
        bitmap += 16 # 2^4
      if inGrid(walls, [xTest, yTest]):
        bitmap += 32 # 2^5*
      matrix.append(bitmap)
  return matrix


class Matrix(MultiAgentSearchAgent):
  def __init__(self):
    self.agent = load('TrainedModels/clfMatrixPlus.joblib')
    lastAction[0] = 1

  def getAction(self, currGameState):
    data = pd.DataFrame(
      columns=dataColumnsDistanceOnly)
    data.loc[0, :] = extractFeatureDistanceOnly(currGameState, lastAction[0])
    dataTrain = data.drop(columns=["labelNextAction"], axis=1)
    nextActionNumber = self.agent.predict(dataTrain)
    nextPredictedAction = getActionByNumber(nextActionNumber)
    lastAction[0] = nextActionNumber

    if (nextPredictedAction not in currGameState.getLegalActions(0)):
      print("Illegal Action")
      nextPredictedAction = currGameState.getLegalActions(0)[random.randrange(0, len(currGameState.getLegalActions(0)))]
    return nextPredictedAction


# Abbreviation
better = betterEvaluationFunction