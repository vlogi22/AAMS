import numpy as np
from abc import ABC, abstractmethod

class BasicAgent(ABC):

  def __init__(self, name: str):
    self.name_ = name
    self.obs_ = None
    self.obsInfo_ = None

  def see(self, observation: np.ndarray, obsInfo :list):
    self.obs_ = observation
    self.obsInfo_ = obsInfo

  @abstractmethod
  def action(self) -> int:
    raise NotImplementedError()
