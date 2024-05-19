import numpy as np
from abc import ABC

R = 0
G = 0
B = 100

class BasicAgent(ABC):

  def __init__(self, agentId: int, name: str, strength: int = 5, speed: int = 5, maxEnergy: int = 100):
    self.agentId_ = agentId
    self.name_ = name
    self.strength_ = strength
    self.speed_ = speed
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
  
  def getMaxEnergy(self) -> int:
    return self.maxEnergy_
  def setMaxEnergy(self, maEnergy: int) -> int:
    self.maxEnergy_ = maEnergy
  
  def getEnergy(self) -> int:
    return self.energy_
  def setEnergy(self, energy) -> int:
    self.energy_ = energy
  def resetEnergy(self):
    self.energy_ = self.maxEnergy_

  def getStrength(self) -> int:
    return self.strength_
  def addStrength(self, strength: int) -> int:
    if (self.strength_ + strength >= 1):
      self.strength_ += strength

  def getSpeed(self) -> int:
    return self.speed_
  def addSpeed(self, speed: int) -> int:
    if (self.speed_ + speed >= 1):
      self.speed_ += speed
    
  def rgbArray(self) -> tuple:
    return (R + 20*self.strength_, G + 20*self.strength_, B)
