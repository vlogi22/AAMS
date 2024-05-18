import numpy as np
from abc import ABC

R = 0
G = 0
B = 100

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

  def getName(self) -> str:
    return self.name_
  def setName(self, name: str):
    self.name_ = name
  
  def getId(self) -> int:
    return self.agentId_
  def setId(self, id: int) -> int:
    self.agentId_ = id
  
  def getMaxEnergy(self) -> float:
    return self.maxEnergy_
  def setMaxEnergy(self, maEnergy: float) -> float:
    self.maxEnergy_ = maEnergy
  
  def getEnergy(self) -> float:
    return self.energy_
  def setEnergy(self, energy) -> float:
    self.energy_ = energy
  def resetEnergy(self):
    self.energy_ = self.maxEnergy_

  def getStrength(self) -> float:
    return self.strength_
  def addStrength(self, strength: float) -> float:
    if (self.strength_ + strength >= 0.1):
      self.strength_ += strength
    
  def rgbArray(self) -> tuple:
    return (R + int(255*self.strength_), G, B)
