import math
import random
import numpy as np
from scipy.spatial.distance import cityblock

from agent.basicAgent import BasicAgent

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

class GreedyAgent(BasicAgent):

  def __init__(self, agentId, strength: float = 0.5, maxEnergy: float = 20):
    super(GreedyAgent, self).__init__(agentId, f"Greedy Agent", strength, maxEnergy)
    self.nActions = N_ACTIONS

  def action(self) -> int:
    agents = self.obsInfo_[0]
    foods = self.obsInfo_[1]

    agentPosition_pos = agents[self.agentId_]
    foodPositions =  list(foods.values())

    closestFoodPos = self.closestFood(agentPosition_pos, foodPositions)
    if closestFoodPos is not None: 
      return self.directionToGo(agentPosition_pos, closestFoodPos)
    else:
      return STAY

  def directionToGo(self, agentPosition, foodPosition):
    """
    Given the position of the agent and the position of a food,
    returns the action to take in order to close the distance
    """
    distances = np.array(foodPosition) - np.array(agentPosition)
    absDistances = np.absolute(distances)
    if absDistances[0] > absDistances[1]:
      return self._closeVertically(distances)
    elif absDistances[0] < absDistances[1]:
      return self._closeHorizontally(distances)
    else:
      roll = random.uniform(0, 1)
      return self._closeHorizontally(distances) if roll > 0.5 else self._closeVertically(distances)

  def closestFood(self, agentPosition, foodPositions):
    """
    Given the positions of an agent and a sequence of positions of all food,
    returns the positions of the closest food.
    If there are no foods, None is returned instead
    """
    min = math.inf
    closestfoodPosition = None
    nFoods = len(foodPositions)

    for i in range(nFoods):
      foodPosition = foodPositions[i]
      distance = cityblock(agentPosition, foodPosition)
      if distance < min:
        min = distance
        closestfoodPosition = foodPosition
    return closestfoodPosition

  def _closeHorizontally(self, distances):
    if distances[1] > 0:
      return RIGHT
    elif distances[1] < 0:
      return LEFT
    else:
      return STAY

  def _closeVertically(self, distances):
    if distances[0] > 0:
      return DOWN
    elif distances[0] < 0:
      return UP
    else:
      return STAY