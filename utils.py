import math
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

def plot(xLen = 0, x = [], xLabel = 'X_Label', yLabel = 'Y_Label', ylim=(-20,20), s=1, image="image", colors=None):
  plt.clf()
  plt.xlabel(xLabel)
  plt.ylabel(yLabel)
  plt.ylim(ylim)
  plt.scatter(np.arange(1, xLen + 1), x, s=s)
  plt.savefig(image, bbox_inches = 'tight')



