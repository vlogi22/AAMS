import numpy as np
from scipy.spatial.distance import cityblock

from agent.basicAgent import BasicAgent

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

class RandomAgent(BasicAgent):

  def __init__(self, agentId):
    super(RandomAgent, self).__init__(agentId, "Random Agent")
    self.nActions = N_ACTIONS

  def action(self) -> int:
    return np.random.randint(self.nActions)