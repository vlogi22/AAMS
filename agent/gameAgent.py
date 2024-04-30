from agent.basicAgent import BasicAgent

import numpy as np
import torch

from dqn import DQN

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

class GameAgent(BasicAgent):

    def __init__(self, agentId, device: str = "cpu"):
      super(GameAgent, self).__init__(f"Greedy Agent")
      self.device_ = device
      self.agentId_ = agentId
      self.nActions_ = N_ACTIONS
      self.brain_ = DQN(device=device)

    def action(self) -> int:
      #print("Qs ", self.brain_.getQs(self.observation/255))
      return np.argmax(self.brain_.getQs(self.observation/255))
    
    def updateReplay(self, obs, action, reward, newObs, done):
      self.brain_.updateReplay(obs, action, reward, newObs, done)

    def train(self, done, step):
      self.brain_.train(done, step)

    def save(self, fileName='model.pth'):
      print("agent", self.agentId_, " saving ", fileName)
      self.brain_.save(fileName)

    def load(self, fileName='model.pth'):
      print("agent", self.agentId_, " loading ", fileName)
      self.brain_.load(fileName)
