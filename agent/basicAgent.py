import numpy as np
from abc import ABC, abstractmethod

class BasicAgent(ABC):

  def __init__(self, name: str):
    self.name_ = name
    self.obs_ = None

  def see(self, observation: np.ndarray):
    self.obs_ = observation

  @abstractmethod
  def action(self) -> int:
    raise NotImplementedError()
