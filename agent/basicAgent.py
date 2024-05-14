import numpy as np
from abc import ABC

class BasicAgent(ABC):

  def __init__(self, agentId: int, name: str, strength: float = 0.5, maxEnergy: float = 20):
    self.agentId_ = agentId
    self.name_ = name
    self.strength_ = strength
    self.maxEnergy_ = maxEnergy
    self.energy_ = maxEnergy
    self.obs_ = None
    self.obsInfo_ = None

  def see(self, observation: np.ndarray, obsInfo :list):
    self.obs_ = observation
    self.obsInfo_ = obsInfo

  def name(self) -> str:
    return self.name_
  
  def id(self) -> int:
    return self.agentId_
  
  def maxEnergy(self) -> float:
    return self.maxEnergy_
  
  def energy(self) -> float:
    return self.energy_
  
  def resetEnergy(self):
    self.energy_ = self.maxEnergy_

  def strength(self) -> float:
    return self.strength_
