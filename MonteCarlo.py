#MCTS implementation found here : https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
#adapted in python 2 for the need of the project and to fit pacman game
import math
from collections import defaultdict
from random import random
from random import randint, shuffle

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError("choose called on terminal node")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            childrens = self.children.keys()
            unexplored = self.children[node].copy()

            for children in childrens:
                unexplored.discard(children)

            # unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)


class Node():
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    def __init__(self, gameState):
        self.gameState = gameState
        self.actions = []

    def find_children(self):
        gameStates = []
        if len(gameStates) > 0:
            gameStateForNextAgent = []
            for gameState in gameStates:
                actions = gameState.getLegalActions(0)
                for action in actions:
                    gameStateForNextAgent.append(gameState.generateSuccessor(0, action))
                for gameStateNextAgent in gameStateForNextAgent:
                    gameStates.append(gameStateNextAgent)
        else :
            actions = self.gameState.getLegalActions(0)
            for action in actions:
                gameStates.append(self.gameState.generateSuccessor(0, action))
        "All possible successors of this board state"
        gameStateNode = []
        for gamestate in gameStates:
            gameStateNode.append(Node(gamestate))
        return set(gameStateNode)

    def find_random_child(self):
        allGameStatePossible =  self.find_children()
        allGameStatePossibleArray = []
        for statePossible in allGameStatePossible:
            allGameStatePossibleArray.append(statePossible)
          
        randomState = allGameStatePossibleArray[randint(0, len(allGameStatePossibleArray) - 1)]
        return randomState

        "Random successor of this board state (for more efficient simulation)"
        return None

    def is_terminal(self):
        return self.gameState.isLose() or self.gameState.isWin()

    def reward(self):
        #TODO : We don't want to only win, we want the biggest score
        if self.gameState.isWin():
            return 1 -   (float(1) / self.gameState.getScore())
        else :
            return 0
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"

    def __hash__(self):
        "Nodes must be hashable"
        return hash(self.gameState.data)

    def __eq__(self, other):
        """
        Allows two states to be compared.
        """
        if other == None: return False
        # TODO Check for type of other
        if not self.gameState.data.agentStates == other.gameState.data.agentStates: return False
        if not self.gameState.data.food == other.gameState.data.food: return False
        if not self.gameState.data.capsules == other.gameState.data.capsules: return False
        if not self.gameState.data.score == other.gameState.data.score: return False
        return True
