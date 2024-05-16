import numpy as np

from agent.greedyDqnAgent import GreedyDQNAgent

class GameAgentFactory:

  def __init__(self, seed = None):
    self.i = 0
    np.random.seed(seed)
  
  def createGreedyDqnAgent(self, strength: float = None, maxEnergy: float = 20, nSpawns: int = 10*10, device: str = "cpu"):
    if (strength == None):
      strength_ = np.random.randint(1, 10)/10
    
    agent = GreedyDQNAgent(self.i, strength_, maxEnergy, nSpawns, device)
    self.i += 1
    return agent
