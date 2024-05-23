import numpy as np

from agent.basicAgent import BasicAgent
from agent.dqn import DQN
from agent.qLearning import QLearning

INC_STRENGTH, DEC_STRENGTH, INC_SPEED, DEC_SPEED, NO_OP = range(5)

class DQNAgent(BasicAgent):

  def __init__(self, agentId, nSpawns, name, 
               strength: int = 5, speed: int = 5, 
               maxEnergy: int = 100, epsilon=1,
               device: str = "cpu", seed: int = None):
    super(DQNAgent, self).__init__(agentId=agentId, name=name, strength=strength, speed=speed, maxEnergy=maxEnergy)
    self.device_ = device
    self.nSpawns_ = nSpawns

    self.spawnBrain_ = DQN(outputLayer = nSpawns, device=device)
    self.epsilonSpawn_ = epsilon
    self.epsilonDecaySpawn_ = 0.99965
    self.minEpsilonSpawn_ = 0.005

    self.geneticBrain_ = QLearning(nActions=5, alpha=0.2, gamma=0.8)
    self.epsilonGenetic_ = epsilon
    self.epsilonDecayGenetic_ = 0.996
    self.minEpsilonGenetic_ = 0.05

    self.seeds_ = seed    # Seed
    np.random.seed(seed)
  
  def seed(self): return self.seeds_

  def epsilonSpawn(self): return self.epsilonSpawn_
  def epsilonDecaySpawn(self): return self.epsilonDecaySpawn_
  def minEpsilonSpawn(self): return self.minEpsilonSpawn_

  def epsilonGenetic(self): return self.epsilonGenetic_
  def epsilonDecayGenetic(self): return self.epsilonDecayGenetic_
  def minEpsilonGenetic(self): return self.minEpsilonGenetic_
  
  def spawnAction(self) -> int:
    if np.random.random() < self.epsilonSpawn_:
      action = np.random.randint(0, self.nSpawns_)
    else:
      action = np.argmax(self.spawnBrain_.getQs(self.obs_/255))
    self.epsilonSpawn_ = max(self.minEpsilonSpawn_, self.epsilonSpawn_*self.epsilonDecaySpawn_)

    return action
  
  def updateSpawnReplay(self, obs, action, reward, newObs, done):
    self.spawnBrain_.updateReplay(obs, action, reward, newObs, done)

  def train(self, done, step):
    return self.spawnBrain_.train(done, step)
  
  def updateGenetic(self, reward: int=None) -> int:
    if np.random.random() < self.epsilonGenetic_:
      action = np.random.randint(0, 5)
    else:
      action = self.geneticBrain_.chooseAction((self.strength_, self.speed_))
    self.epsilonGenetic_ = max(self.minEpsilonGenetic_, self.epsilonGenetic_*self.epsilonDecayGenetic_)

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
    
    if (reward == None): return action

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
