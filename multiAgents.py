# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "* YOUR CODE HERE *"
        score = successorGameState.getScore()

        # Xử lý điểm dựa trên khoảng cách hiện tại với food
        foodList = newFood.asList()
        if foodList:
            minFoodDistance = min([manhattanDistance(newPos, food) for food in foodList])
            if minFoodDistance == 0:
                score += 10.0
            else:
                score += 10.0 / minFoodDistance

        # Xử lý điểm dựa trên khoảng cách hiện tại với ghost
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            distanceToGhost = manhattanDistance(newPos, ghostPos)

            if distanceToGhost == 0:
                if scaredTime > 0.5:
                    score += 10
                else:
                    score -= 500
            else:
                if scaredTime > 0.5:
                    score += 10.0 / distanceToGhost
                else:
                    score -= 10.0 / distanceToGhost

        # Phạt điểm nặng nếu dừng
        if action == Directions.STOP:
            score -= 50 

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "* YOUR CODE HERE *"
        def minimax(agentIndex, depth, gameState):
            # Trước tiên check xem game đã hết hay đã tới max depth chưa
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman là maximizer có index = 0
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)
            # Ghosts là minimizer có index = 1 hoặc cao hơn
            else:
                return minValue(agentIndex, depth, gameState)
        
        def maxValue(agentIndex, depth, gameState):
            # khởi tạo 1 giá trị điểm minimax siêu thấp
            v = float('-inf')
            # Loop lấy actions
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)

            # Với từng action thì gen ra GameState mới, tìm giá trị max mới
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                # đặt index 1 vì hết lượt pacman lúc nào cx là ghost 1 tiếp
                v = max(v, minimax(1, depth, successor)) 
            return v

        def minValue(agentIndex, depth, gameState):
            # khởi tạo 1 giá trị điểm minimax siêu cao
            v = float('inf')
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)

            # Loop qua tất cả các ghost để tìm action đối phó tốt nhất của nó
            nextAgent = agentIndex + 1
            # trở lại lượt pacman khi đã qua hết ghost sau đó tăng depth
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0
                depth += 1
            
            # làm tương tự Pacman nhưng ghost thì lấy điểm min
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, minimax(nextAgent, depth, successor))
            return v

        # Sau khi có điểm cho từng action hợp lệ của pacman thì chọn action có điểm cao nhất
        bestAction = None
        bestScore = float('-inf')

        for action in gameState.getLegalActions(0):  # Loop các actions của agent 0 - pacman
            successor = gameState.generateSuccessor(0, action)
            score = minimax(1, 0, successor)  # Gọi hàm minimax để lấy điểm từ ghost 1, depth 0
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(agentIndex, depth, gameState, alpha, beta):
            # Base case: Check game kết thúc hoặc đã tới max depth chưa
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman (maximizer) có agentIndex = 0
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState, alpha, beta)
            # Ghosts (minimizer) có agentIndex >= 1
            else:
                return minValue(agentIndex, depth, gameState, alpha, beta)

        def maxValue(agentIndex, depth, gameState, alpha, beta):
            v = float('-inf')
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)

            # Loop qua tất cả các action hợp lệ và cập nhật điểm 
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v, alphabeta(1, depth, successor, alpha, beta)) 
                if v > beta:
                    return v  # Prune
                alpha = max(alpha, v)
            return v

        def minValue(agentIndex, depth, gameState, alpha, beta):
            v = float('inf')
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)

            # Tăng agent index, tăng depth khi hết ghost
            nextAgent = agentIndex + 1
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0 
                depth += 1  

            # Loop qua tất cả các action hợp lệ và cập nhật điểm 
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, alphabeta(nextAgent, depth, successor, alpha, beta))
                if v < alpha:
                    return v  # Prune
                beta = min(beta, v)
            return v

        # Pacman (agentIndex = 0) chọn action có điểm Alpha-Beta cao nhất
        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions(0):  # Loop các action hợp lệ của pacman
            successor = gameState.generateSuccessor(0, action)
            score = alphabeta(1, 0, successor, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, score)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(gameState, agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return maxValue(gameState, agentIndex, depth)
            else:
                return expValue(gameState, agentIndex, depth)

        def maxValue(gameState, agentIndex, depth):
            v = float("-inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v,expectimax(successor, 1, depth))
            return v

        def expValue(gameState, agentIndex, depth):
            v = 0.0
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                next_agentIndex = (agentIndex + 1) % gameState.getNumAgents()
                next_depth = depth - 1 if next_agentIndex == 0 else depth
                v += expectimax(successor, next_agentIndex, next_depth) / len(gameState.getLegalActions(agentIndex))
            return v
        
        max_score = float("-inf")
        for action in gameState.getLegalActions(0):
            score = expectimax(gameState.generateSuccessor(0, action), 1, self.depth)
            if score > max_score:
                max_score, best_action = score, action
        return best_action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: it is better :)
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    ghostScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    evaluation = 0

    if food:
        minFoodDistance = min(manhattanDistance(pacmanPos, foodPos) for foodPos in food)
        evaluation += 1.0 / (minFoodDistance + 1) 

    for ghostState, scaredTime in zip(ghostStates, ghostScaredTimes):
        ghostPos = ghostState.getPosition()
        ghostDistance = manhattanDistance(pacmanPos, ghostPos)

        if scaredTime > 0.5:
            evaluation += 200.0 / (ghostDistance + 1)  
        else:
            if ghostDistance <= 2:
                evaluation -= 100.0  # Trừ nhiều điểm nếu ghost quá gần

    evaluation -= 10 * len(food)  # Trừ càng nhiều điểm với càng nhiều food còn lại

    capsules = currentGameState.getCapsules()
    if capsules:
        minCapsuleDistance = min(manhattanDistance(pacmanPos, capsulePos) for capsulePos in capsules)
        evaluation += 10 / (minCapsuleDistance + 1)

    return evaluation

# Abbreviation
better = betterEvaluationFunction
