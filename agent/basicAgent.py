import numpy as np
from abc import ABC

class BasicAgent(ABC):

  def __init__(self, agentId: int, name: str):
    self.agentId_ = agentId
    self.name_ = name
    self.obs_ = None
    self.obsInfo_ = None

  def see(self, observation: np.ndarray, obsInfo :list):
    self.obs_ = observation
    self.obsInfo_ = obsInfo

  def name(self) -> str:
    return self.name_
  
  def id(self) -> int:
    return self.agentId_
