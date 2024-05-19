import numpy as np

from agent.basicAgent import BasicAgent
from agent.dqn import DQN
from agent.qLearning import QLearning

INC_STRENGTH, DEC_STRENGTH = range(2)

class DQNAgent(BasicAgent):

  def __init__(self, agentId, nSpawns, nGenetics, name, strength: float = 0.5, maxEnergy: float = 20, device: str = "cpu"):
    super(DQNAgent, self).__init__(agentId=agentId, name=name, strength=strength, maxEnergy=maxEnergy)
    self.device_ = device
    self.nSpawns_ = nSpawns
    self.spawnBrain_ = DQN(outputLayer = nSpawns, device=device)
    self.geneticBrain_ = QLearning(nActions=nGenetics, minEpsilon=0.05, alpha=0.2, gamma=0.9)

  def spawnAction(self) -> int:
    #print("Qs ", self.spawnBrain_.getQs(self.obs_/255))
    return np.argmax(self.spawnBrain_.getQs(self.obs_/255))
  
  def updateSpawnReplay(self, obs, action, reward, newObs, done):
    self.spawnBrain_.updateReplay(obs, action, reward, newObs, done)

  def train(self, done, step):
    return self.spawnBrain_.train(done, step)
  
  def updateGenetic(self, reward: int) -> int:
    action = self.geneticBrain_.chooseAction((self.strength_,))
    strength = self.getStrength()
    
    if action == INC_STRENGTH:
      self.addStrength(0.1)
    elif action == DEC_STRENGTH:
      self.addStrength(-0.1)
    else:
      raise Exception('Genetic action does not exists!')
    
    newStrength = self.getStrength()
    self.geneticBrain_.learnQ((strength,), action, reward, (newStrength,))

  def save(self, prefix='model'):
    print("agent", self.agentId_, " saving ", prefix)
    self.spawnBrain_.save(f"{prefix}Spawn")
    self.geneticBrain_.save(f"{prefix}Genetic")

  def load(self, prefix='model'):
    print("agent", self.agentId_, " loading ", prefix)
    self.spawnBrain_.load(f"{prefix}Spawn")
    self.geneticBrain_.load(f"{prefix}Genetic")
