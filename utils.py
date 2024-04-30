import math
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

def plot(n_ep, scores, filename="image", colors=None):
    plt.clf()
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.plot(np.arange(1, n_ep + 1), scores)
    plt.savefig(filename, bbox_inches = 'tight')


