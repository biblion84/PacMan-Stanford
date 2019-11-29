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

dataColumns = ["ghostUp", "ghostDown", "ghostLeft", "ghostRight", "wallUp", "wallDown", "wallLeft", "wallRight",
               "foodUp", "foodDown", "foodLeft", "foodRight", "emptyUp", "emptyDown", "emptyLeft", "emptyRight",
               "nearestFood", "nearestGhost", "nearestCapsule", "legalPositionUp", "legalPositionDown",
               "legalPositionULeft", "legalPositionRight", "pacmanPositionX", "pacmanPositionY", "lastAction", "labelNextAction"]

def scoreEvaluationFunction(currentGameState):
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)
    self.indexDataframe = 0
    self.alreadyWroteHeaders = False
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
  if len(currGameState.getLegalActions(0)) == 2:
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

lastAction = "None"
AlreadyVisited = []
HistoActions = []

# Abbreviation
better = betterEvaluationFunction