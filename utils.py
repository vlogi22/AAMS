import math
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

def plot(x=[], y=[], xLabel='X_Label', yLabel='Y_Label', xlim=None, ylim=None, s=1, image="image", colors=None):
  plt.clf()
  plt.xlabel(xLabel)
  plt.ylabel(yLabel)
  if xlim != None:
    plt.xlim(xlim)
  if ylim != None:
    plt.ylim(ylim)
  plt.scatter(x=x, y=y, s=s)
  plt.savefig(image, bbox_inches = 'tight')



