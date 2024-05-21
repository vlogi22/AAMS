import numpy as np

from agent.basicAgent import BasicAgent
from agent.dqn import DQN
from agent.qLearning import QLearning

INC_STRENGTH, DEC_STRENGTH, INC_SPEED, DEC_SPEED, NO_OP = range(5)

class DQNAgent(BasicAgent):

  def __init__(self, agentId, nSpawns, name, 
               strength: int = 5, speed: int = 5, 
               maxEnergy: int = 100, device: str = "cpu"):
    super(DQNAgent, self).__init__(agentId=agentId, name=name, strength=strength, speed=speed, maxEnergy=maxEnergy)
    self.device_ = device
    self.nSpawns_ = nSpawns
    self.spawnBrain_ = DQN(outputLayer = nSpawns, device=device)
    self.geneticBrain_ = QLearning(nActions=5, minEpsilon=0.05, alpha=0.2, gamma=0.9)

  def spawnAction(self) -> int:
    #print("Qs ", self.spawnBrain_.getQs(self.obs_/255))
    return np.argmax(self.spawnBrain_.getQs(self.obs_/255))
  
  def updateSpawnReplay(self, obs, action, reward, newObs, done):
    self.spawnBrain_.updateReplay(obs, action, reward, newObs, done)

  def train(self, done, step):
    return self.spawnBrain_.train(done, step)
  
  def updateGenetic(self, reward: int) -> int:
    action = self.geneticBrain_.chooseAction((self.strength_, self.speed_))
    strength = self.getStrength()
    speed = self.getSpeed()
    
    if action == INC_STRENGTH:
      self.addStrength(1)
    elif action == DEC_STRENGTH:
      self.addStrength(-1)
    elif action == INC_SPEED:
      self.addSpeed(1)
    elif action == DEC_SPEED:
      self.addSpeed(-1)
    
    newStrength = self.getStrength()
    newSpeed = self.getSpeed()
    self.geneticBrain_.learnQ((strength, speed), action, reward, (newStrength, newSpeed))

    return action

  def save(self, prefix='model'):
    print("agent", self.agentId_, " saving ", prefix)
    self.spawnBrain_.save(f"{prefix}Spawn")
    self.geneticBrain_.save(f"{prefix}Genetic")

  def load(self, prefix='model'):
    print("agent", self.agentId_, " loading ", prefix)
    self.spawnBrain_.load(f"{prefix}Spawn")
    self.geneticBrain_.load(f"{prefix}Genetic")
