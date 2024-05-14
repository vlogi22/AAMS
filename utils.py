import math
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

def plot(n_ep, scores, ylim=(-20,20), image="image", colors=None):
  plt.clf()
  plt.xlabel('Episodes')
  plt.ylabel('Score')
  plt.ylim(ylim)
  plt.scatter(np.arange(1, n_ep + 1), scores, s=0.1)
  plt.savefig(image, bbox_inches = 'tight')



