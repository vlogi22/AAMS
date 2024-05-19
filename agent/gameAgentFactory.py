import numpy as np

from agent.greedyDqnAgent import GreedyDQNAgent

class GameAgentFactory:

  def __init__(self, seed = None):
    self.i = 0
    np.random.seed(seed)
  
  def createGreedyDqnAgent(self, strength: int = None, speed: int = None, maxEnergy: int = 100, 
                           nSpawns: int = 10*10, device: str = "cpu"):
    if (strength == None):
      strength = np.random.randint(1, 11)
    if (speed == None):
      speed = np.random.randint(1, 11)
    
    agent = GreedyDQNAgent(agentId=self.i, strength=strength, speed=speed, maxEnergy=maxEnergy, 
                           nSpawns=nSpawns, device=device)
    self.i += 1
    return agent
