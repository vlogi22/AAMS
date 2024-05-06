import numpy as np
from abc import ABC, abstractmethod

class BasicAgent(ABC):

  def __init__(self, name: str):
    self.name = name
    self.observation = None

  def see(self, observation: np.ndarray):
    self.observation = observation

  @abstractmethod
  def action(self) -> int:
    raise NotImplementedError()
