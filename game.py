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
  d = {'player': (255, 0, 0),
        'food': (0, 255, 0),
        'enemy': (0, 0, 255)}
  
  metadata = {'render.modes': ['human', 'rgb_array']}

  def __init__(self, gridShape=(10, 10), nFoods=1,
              penalty=-2, stepCost=-1, foodCaptureReward=5, maxSteps=100):
    
    # Game Args
    self.gridShape_ = gridShape
    self.nAgents_ = 0
    self.nFoods_ = nFoods
    self.maxSteps_ = maxSteps
    self.stepCount_ = 0

    # Scores
    self.penalty_ = penalty
    self.stepCost_ = stepCost
    self.foodCaptureReward_ = foodCaptureReward

    # Game objects
    self.foodPos_ = {}
    self.agentPos_ = {}
    self.agents_ = {}
    self.agentDones_ = {}

    # Game Map View
    self.baseGrid_ = self.__create_grid()  # with no agents
    self.fullObs_ = [[{PRE_IDS['agent']: [], PRE_IDS['food']: []} for _ in range(self.gridShape_[1])] for _ in range(self.gridShape_[0])]
    self.viewer_ = None

    self.seed()

  def getGridShape(self):
    return self.gridShape_

  def seed(self, n=None):
    self.np_random, seed = seeding.np_random(n)
    return [seed]
  
  def reset(self, nFoods = 20, pattern = []):
    # Game Args
    self.stepCount_ = 0
    self.nFoods_ = nFoods

    # Scores
    self.foodPos_ = {}
    self.agentPos_ = {}
    self.agentDones_ = {}

    # Game Map View
    if len(pattern):
      self.__init_full_obs_pattern(pattern)
    else:
      self.__init_full_obs()

    return [self.get_agent_obs(agentId) for agentId in range(0, self.nAgents_)], [self.agentPos_, self.foodPos_]
  
  def getAgent(self, agentId: int):
    return self.agents_[agentId]

  def addAgent(self, agentId: int, agent):
    self.agents_[agentId] = agent
    self.nAgents_ += 1

  def spawn(self, agent, pos):
    reward = abs(self.agentPos_[agent.id()][0] - pos[0]) + abs(self.agentPos_[agent.id()][1] - pos[1])
    self.agentPos_[agent.id()] = pos
    self.__update_agent_view(agent.id())
    return -reward/4
  
  def step(self, agents_action: dict):
    self.stepCount_ += 1
    rewards = [self.stepCost_ for _ in range(self.nAgents_)]

    for agent_i, action in agents_action.items():
      if not (self.agentDones_[agent_i]):
        # After a move, it will return a additional reward value if something happen
        rewards[agent_i] += self.__update_agent_pos(agent_i, action)
        
    rewards = self.__check_colisions(rewards)

    for agent_i, agent in self.agents_.items():
      self.agentDones_[agent.id()] = not agent.canMove()

    if (self.stepCount_ >= self.maxSteps_) or (not self.foodPos_):
      for i in range(self.nAgents_):
        self.agentDones_[i] = True

    return [self.get_agent_obs(agentId) for agentId in range(0, self.nAgents_)], \
              [self.agentPos_, self.foodPos_], rewards, self.agentDones_.values()

  def get_agent_obs(self, agentId):
    env = np.zeros((3, self.gridShape_[0], self.gridShape_[1]), dtype=np.float32)
    
    for _, [row, col] in self.foodPos_.items():
      # sets the food location tile to it's color
      env[0][row][col] = self.d['food'][0]
      env[1][row][col] = self.d['food'][1]
      env[2][row][col] = self.d['food'][2]

    for id, [row, col] in self.agentPos_.items():
      rgb = self.agents_[id].rgbArray()
      # sets the enemy location tile to it's color
      env[0][row][col] = rgb[0]
      env[1][row][col] = rgb[1]
      env[2][row][col] = rgb[2]

    # sets the player location tile to it's color
    rgb = self.agents_[agentId].rgbArray()
    [row, col] = self.agentPos_[agentId]
    env[0][row][col] = rgb[0]
    env[1][row][col] = rgb[1]
    env[2][row][col] = rgb[2] + 100
    
    return env

  def __init_full_obs_pattern(self, pat: list):
    self.fullObs_ = [[{PRE_IDS['agent']: [], PRE_IDS['food']: []} for _ in range(self.gridShape_[1])] for _ in range(self.gridShape_[0])]

    for agent_i, _ in self.agents_.items():
      pos = [-1, -1]
      while not self._is_cell_vacant(pos):
        pos = [self.np_random.randint(0, self.gridShape_[0] - 1),
                self.np_random.randint(0, self.gridShape_[1] - 1)]
      self.agentPos_[agent_i] = pos
      # Add the agent to the grid
      self.__update_agent_view(agent_i)
    self.agentDones_ = {agent_i:False for agent_i, _ in self.agents_.items()}

    for food_i in range(self.nFoods_):
      pos = [-1, -1]
      while not self._is_cell_vacant(pos):
        pos = [self.np_random.randint(0, self.gridShape_[0] - 1),
                self.np_random.randint(0, self.gridShape_[1] - 1)]
        
        ran = self.np_random.uniform(0, 1)

        if (ran < pat[pos[0]][pos[1]]):
          self.foodPos_[food_i] = pos
        else:
          pos = [-1, -1]
      # Add the food to the grid
      self.__update_food_view(food_i)

  def __init_full_obs(self):
    self.fullObs_ = [[{PRE_IDS['agent']: [], PRE_IDS['food']: []} for _ in range(self.gridShape_[1])] for _ in range(self.gridShape_[0])]

    for agent_i, _ in self.agents_.items():
      pos = [-1, -1]
      while not self._is_cell_vacant(pos):
        pos = [self.np_random.randint(0, self.gridShape_[0] - 1),
                self.np_random.randint(0, self.gridShape_[1] - 1)]
      self.agentPos_[agent_i] = pos
      # Add the agent to the grid
      self.__update_agent_view(agent_i)
    self.agentDones_ = {agent_i:False for agent_i, _ in self.agents_.items()}

    for food_i in range(self.nFoods_):
      pos = [-1, -1]
      while not self._is_cell_vacant(pos):
        pos = [self.np_random.randint(0, self.gridShape_[0] - 1),
                self.np_random.randint(0, self.gridShape_[1] - 1)]
      self.foodPos_[food_i] = pos
      # Add the food to the grid
      self.__update_food_view(food_i)

  def is_valid(self, pos):
    return (0 <= pos[0] < self.gridShape_[0]) and (0 <= pos[1] < self.gridShape_[1])

  def _is_cell_vacant(self, pos):
    return self.is_valid(pos) and (len(self.fullObs_[pos[0]][pos[1]][PRE_IDS['agent']]) == 0) and (len(self.fullObs_[pos[0]][pos[1]][PRE_IDS['food']]) == 0)

#
# Grid
#

  def __create_grid(self):
    return [[PRE_IDS['empty'] for _ in range(self.gridShape_[1])] for _ in range(self.gridShape_[0])]
  
  def __update_agent_view(self, agent_i):
    [row, col] = self.agentPos_[agent_i]
    self.fullObs_[row][col][PRE_IDS['agent']].append(agent_i)

  def __update_food_view(self, food_i):
    [row, col] = self.foodPos_[food_i]
    self.fullObs_[row][col][PRE_IDS['food']].append(food_i)

  def __update_agent_pos(self, agentId, move):
    curr_pos = copy.copy(self.agentPos_[agentId])
    next_pos = self.__next_pos(curr_pos, move)

    reward = 0
    
    if self.is_valid(next_pos):
      self.agentPos_[agentId] = next_pos
      self.fullObs_[curr_pos[0]][curr_pos[1]][PRE_IDS['agent']].remove(agentId)
      self.__update_agent_view(agentId)
    else:
       # An invalid move, i.e., out of bound.
       reward += self.penalty_

    return reward
  
  def __check_colisions(self, rewards):
    for x in range(self.gridShape_[0]):
        for y in range(self.gridShape_[1]):
          agents_at_position = self.fullObs_[x][y][self.PRE_IDS['agent']]
          foods_at_position = self.fullObs_[x][y][self.PRE_IDS['food']]
          
          if len(foods_at_position) > 0:
            if len(agents_at_position) == 1:
              rewards[agents_at_position[0]] += self.foodCaptureReward_
              self.foodPos_.pop(foods_at_position[0])
              self.fullObs_[x][y][PRE_IDS['food']].remove(foods_at_position[0])
              
            elif len(agents_at_position) > 1:
              total_strength = sum([self.agents_[agent_id].strength() for agent_id in agents_at_position])
              reward_probabilities = [self.agents_[agent_id].strength() / total_strength for agent_id in agents_at_position]
              chosen_agent_id = random.choices(agents_at_position, weights=reward_probabilities)[0]
              rewards[chosen_agent_id] += self.foodCaptureReward_
              self.foodPos_.pop(foods_at_position[0])
              self.fullObs_[x][y][PRE_IDS['food']].remove(foods_at_position[0])
    
    return rewards
              
    

  def __next_pos(self, curr_pos, move):
    if move == 0:  # down
      next_pos = [curr_pos[0] + 1, curr_pos[1]]
    elif move == 1:  # left
      next_pos = [curr_pos[0], curr_pos[1] - 1]
    elif move == 2:  # up
      next_pos = [curr_pos[0] - 1, curr_pos[1]]
    elif move == 3:  # right
      next_pos = [curr_pos[0], curr_pos[1] + 1]
    elif move == 4:  # stay
      next_pos = curr_pos
    else:
      raise Exception('Action Not found!')
    return next_pos
  
#
# Rendering
#
    
  def render(self, mode='human'):
    img = draw_grid(self.gridShape_[0], self.gridShape_[1], cell_size=CELL_SIZE, fill='white')

    for agent_i, pos in self.agentPos_.items():
      agentRGB = self.agents_[agent_i].rgbArray()
      draw_circle(img, pos, cell_size=CELL_SIZE, fill=agentRGB)
      write_cell_text(img, text=str(agent_i), pos=pos, cell_size=CELL_SIZE,
                      fill='white', margin=0.4)

    for food_i, pos in self.foodPos_.items():
      draw_circle(img, pos, cell_size=CELL_SIZE, fill=FOOD_COLOR)
      write_cell_text(img, text=str(food_i), pos=pos, cell_size=CELL_SIZE,
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

  def close(self):
    if self.viewer_ is not None:
      self.viewer_.close()
      self.viewer_ = None


FOOD_COLOR = (0, 255, 0)

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
