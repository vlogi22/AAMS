import numpy as np

from agent.basicAgent import BasicAgent
from agent.dqn import DQN

class DQNAgent(BasicAgent):

  def __init__(self, agentId, device: str = "cpu"):
    super(DQNAgent, self).__init__(f"DQNAgent")
    self.device_ = device
    self.agentId_ = agentId
    self.nActions_ = 4
    self.brain_ = DQN(outputLayer = self.nActions_, device=device)

  def action(self) -> int:
    #print("Qs ", self.brain_.getQs(self.observation/255))
    return np.argmax(self.brain_.getQs(self.observation/255))
  
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
