import numpy as np

from agent.greedyDqnAgent import GreedyDQNAgent

class GameAgentFactory:

  def __init__(self, seed = None):
    self.i = 0
    np.random.seed(seed)
  
  def createGreedyDqnAgent(self, strength: float = None, maxEnergy: float = 100, 
                           nSpawns: int = 10*10, nGenetics: int = 2, device: str = "cpu"):
    if (strength == None):
      strength = np.random.randint(1, 11)
    
    agent = GreedyDQNAgent(agentId=self.i, strength=strength, maxEnergy=maxEnergy, 
                           nSpawns=nSpawns, nGenetics=nGenetics, device=device)
    self.i += 1
    return agent
