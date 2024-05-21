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

def plot3d(x=[], y=[], z=[], xLabel='X_Label', yLabel='Y_Label', zLabel='Z_Label', 
         xlim=None, ylim=None, zlim=None, s=1, image="image", colors=None):
  plt.clf()
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlabel(xLabel)
  ax.set_ylabel(yLabel)
  ax.set_zlabel(zLabel)

  if xlim is None:
    xlim = (min(x), max(x))
  if ylim is None:
    ylim = (min(y), max(y))
  if zlim is None:
    zlim = (min(z), max(z))

  ax.set_xlim(xlim)
  ax.set_ylim(ylim)
  ax.set_zlim(zlim)

  scatter = ax.scatter(x, y, z, s=s, c=colors)
  
  for i in range(len(x)):
    ax.plot([x[i], x[i]], [y[i], y[i]], [zlim[0], z[i]], color='gray', linestyle='--', linewidth=0.5)
    ax.plot([x[i], xlim[0]], [y[i], y[i]], [z[i], z[i]], color='gray', linestyle='--', linewidth=0.5)
    ax.plot([x[i], x[i]], [ylim[0], y[i]], [z[i], z[i]], color='gray', linestyle='--', linewidth=0.5)
      
  ax.view_init(elev=30, azim=30)
  plt.savefig(image)

def plotHeatMap(matrix, xLabel='X_Label', yLabel='Y_Label', 
                title='Heatmap', cmap='viridis', 
                colorbar_label='Value', image='matrix_heatmap.png'):
  plt.clf()
  fig, ax = plt.subplots()
  cax = ax.matshow(matrix, cmap=cmap)
  
  # Add color bar
  cbar = fig.colorbar(cax)
  cbar.set_label(colorbar_label)
  
  # Set labels
  ax.set_xlabel(xLabel)
  ax.set_ylabel(yLabel)
  plt.title(title)
  
  # Save the figure
  plt.savefig(image, bbox_inches='tight')