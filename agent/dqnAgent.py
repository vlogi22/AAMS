import numpy as np

from agent.basicAgent import BasicAgent
from agent.dqn import DQN

class DQNAgent(BasicAgent):

  def __init__(self, agentId, nSpawns, name, device: str = "cpu"):
    super(DQNAgent, self).__init__(agentId, name)
    self.device_ = device
    self.nSpawns_ = nSpawns
    self.brain_ = DQN(outputLayer = nSpawns, device=device)

  def spawnAction(self) -> int:
    #print("Qs ", self.brain_.getQs(self.obs_/255))
    return np.argmax(self.brain_.getQs(self.obs_/255))
  
  def updateReplay(self, obs, action, reward, newObs, done):
    self.brain_.updateReplay(obs, action, reward, newObs, done)

  def train(self, done, step):
    return self.brain_.train(done, step)

  def save(self, fileName='model.pth'):
    print("agent", self.agentId_, " saving ", fileName)
    self.brain_.save(fileName)

  def load(self, fileName='model.pth'):
    print("agent", self.agentId_, " loading ", fileName)
    self.brain_.load(fileName)
