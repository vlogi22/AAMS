import copy
import logging
import random

import numpy as np

logger = logging.getLogger(__name__)

from PIL import ImageColor, Image
import gym
from gym.utils import seeding

from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text

class Game(gym.Env):
  d = {'player': (255, 175, 0),
        'food': (0, 255, 0),
        'enemy': (0, 0, 255)}
  
  metadata = {'render.modes': ['human', 'rgb_array']}

  def __init__(self, gridShape=(10, 10), nAgents=1, nFoods=1,
              penalty=-2, stepCost=-1, foodCaptureReward=50, maxSteps=200):
    self.gridShape_ = gridShape
    self.nAgents_ = nAgents
    self.nFoods_ = nFoods
    self.maxSteps_ = maxSteps
    self.stepCount_ = None
    self.penalty_ = penalty
    self.stepCost_ = stepCost
    self.foodCaptureReward_ = foodCaptureReward

    self.agentPos_ = {_: None for _ in range(self.nAgents_)}
    self.foodPos_ = {_: None for _ in range(self.nFoods_)}
    self.foodExists_ = {id:True for id in range(self.nFoods_)}
    self.agentDones_ = {id:False for id in range(self.nAgents_)}

    self.baseGrid_ = self.__create_grid()  # with no agents
    self.fullObs_ = self.__create_grid()
    self.viewer_ = None

    self.seed()

  def reset(self, nFoods = 20):
    self.agentPos_ = {}
    self.foodPos_ = {}
    self.nFoods_ = nFoods

    self.__init_full_obs()
    self.stepCount_ = 0
    self.agentDones_ = {id:False for id in range(self.nAgents_)}
    self.foodExists_ = {id:True for id in range(self.nFoods_)}

    return [self.__get_agent_obs_rgb(agentId) for agentId in range(0, self.nAgents_)]

  def step(self, agents_action):
    self.stepCount_ += 1
    rewards = [self.stepCost_ for _ in range(self.nAgents_)]

    for agent_i, action in enumerate(agents_action):
      if not (self.agentDones_[agent_i]):
        # After a move, it will return a additional reward value if something happen
        rewards[agent_i] += self.__update_agent_pos(agent_i, action)

    if (self.stepCount_ >= self.maxSteps_) or (True not in self.foodExists_.values()):
 
      for i in range(self.nAgents_):
        self.agentDones_[i] = True

    return [self.__get_agent_obs_rgb(agentId) for agentId in range(0, self.nAgents_)], \
                rewards, self.agentDones_.values()

  ###
  def __draw_base_img(self):
    self._base_img = draw_grid(self.gridShape_[0], self.gridShape_[1], cell_size=CELL_SIZE, fill='white')
  
  def __get_agent_obs_rgb(self, agentId):
    env = np.zeros((3, self.gridShape_[0], self.gridShape_[1]), dtype=np.float32)  # starts an rbg of our size
    
    for _, [x, y] in self.foodPos_.items():
      # sets the food location tile to it's color
      env[0][x][y] = self.d['food'][0]
      env[1][x][y] = self.d['food'][1]
      env[2][x][y] = self.d['food'][2]

    for _, [x, y] in self.agentPos_.items():
      # sets the enemy location tile to it's color
      env[0][x][y] = self.d['enemy'][0]
      env[1][x][y] = self.d['enemy'][1]
      env[2][x][y] = self.d['enemy'][2]

    # sets the player location tile to it's color
    [x, y] = self.agentPos_[agentId]
    env[0][x][y] = self.d['player'][0]
    env[1][x][y] = self.d['player'][1]
    env[2][x][y] = self.d['player'][2]
    
    return env
    
  ###
  def __create_grid(self):
    _grid = [[PRE_IDS['empty'] for _ in range(self.gridShape_[1])] for _ in range(self.gridShape_[0])]
    return _grid

  ###
  def __init_full_obs(self):
    self.fullObs_ = self.__create_grid()

    for agent_i in range(self.nAgents_):
      while True:
        pos = [self.np_random.randint(0, self.gridShape_[0] - 1),
                self.np_random.randint(0, self.gridShape_[1] - 1)]
        if self._is_cell_vacant(pos):
          self.agentPos_[agent_i] = pos
          break
      # Add the agent to the grid
      self.__update_agent_view(agent_i)

    for food_i in range(self.nFoods_):
      while True:
        pos = [self.np_random.randint(0, self.gridShape_[0] - 1),
                self.np_random.randint(0, self.gridShape_[1] - 1)]
        if self._is_cell_vacant(pos):
          self.foodPos_[food_i] = pos
          break
      # Add the food to the grid
      self.__update_food_view(food_i)

    self.__draw_base_img()

  ###
  def is_valid(self, pos):
    return (0 <= pos[0] < self.gridShape_[0]) and (0 <= pos[1] < self.gridShape_[1])

  ###
  def _is_cell_vacant(self, pos):
    return self.is_valid(pos) and (self.fullObs_[pos[0]][pos[1]] == PRE_IDS['empty'])

  ###
  def __update_agent_pos(self, agentId, move):
    curr_pos = copy.copy(self.agentPos_[agentId])
    next_pos = self.__next_pos(curr_pos, move)

    reward = 0
    foodId = -1
    
    if self._is_cell_vacant(next_pos):
      self.agentPos_[agentId] = next_pos
      self.fullObs_[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
      self.__update_agent_view(agentId)
    elif self.is_valid(next_pos):
      # Check colisions
      self.agentPos_[agentId] = next_pos
      for id, pos in self.foodPos_.items():
         if pos == next_pos:
            foodId = id
            reward += self.foodCaptureReward_
    else:
       # An invalid move, i.e., out of bound.
       reward += self.penalty_

    if foodId != -1:
      self.nFoods_ -= 1
      self.foodPos_.pop(foodId)
      self.foodExists_.pop(foodId)
    return reward

  def __next_pos(self, curr_pos, move):
    if move == 0:  # down
      next_pos = [curr_pos[0] + 1, curr_pos[1]]
    elif move == 1:  # left
      next_pos = [curr_pos[0], curr_pos[1] - 1]
    elif move == 2:  # up
      next_pos = [curr_pos[0] - 1, curr_pos[1]]
    elif move == 3:  # right
      next_pos = [curr_pos[0], curr_pos[1] + 1]
    else:
      raise Exception('Action Not found!')
    return next_pos
  ###
  def __update_agent_view(self, agent_i):
    self.fullObs_[self.agentPos_[agent_i][0]][self.agentPos_[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)
  ###
  def __update_food_view(self, food_i):
    self.fullObs_[self.foodPos_[food_i][0]][self.foodPos_[food_i][1]] = PRE_IDS['food'] + str(food_i + 1)

  def render(self, mode='human'):
    img = copy.copy(self._base_img)

    for agent_i in self.agentPos_.keys():
      draw_circle(img, self.agentPos_[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
      write_cell_text(img, text=str(agent_i), pos=self.agentPos_[agent_i], cell_size=CELL_SIZE,
                      fill='white', margin=0.4)

    for food_i in self.foodPos_.keys():
      if self.foodExists_[food_i]:
          draw_circle(img, self.foodPos_[food_i], cell_size=CELL_SIZE, fill=FOOD_COLOR)
          write_cell_text(img, text=str(food_i), pos=self.foodPos_[food_i], cell_size=CELL_SIZE,
                          fill='white', margin=0.4)

    img = np.asarray(img)
    if mode == 'rgb_array':
      return img
    elif mode == 'human':
      from gym.envs.classic_control import rendering
      if self.viewer_ is None:
        self.viewer_ = rendering.SimpleImageViewer()
      self.viewer_.imshow(img)
      return self.viewer_.isopen

  ###
  def seed(self, n=None):
    self.np_random, seed = seeding.np_random(n)
    return [seed]

  ###
  def close(self):
    if self.viewer_ is not None:
      self.viewer_.close()
      self.viewer_ = None


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
FOOD_COLOR = 'red'

CELL_SIZE = 35

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
}

PRE_IDS = {
    'agent': 'A',
    'food': 'P',
    'empty': '0'
}
