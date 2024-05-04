import numpy as np
from gym import Env
from typing import Sequence
import argparse

import torch
from agent.basicAgent import BasicAgent
from agent.gameAgent import GameAgent
from utils import plot
from game import Game
import time

EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.005

def train_multi_agent(env: Env, agents: Sequence[BasicAgent], n_foods, n_eps: int) -> np.ndarray:
  epsilon = 1
  ep_rewards = []

  for ep in range(n_eps):
    if not (ep % 500):
      print("ep: ", ep, "epsilon", epsilon)
    step = 0
    terminals = [False for _ in range(len(agents))]
    ep_reward = np.zeros(len(agents))
    obs = env.reset(n_foods)
    #env.render()

    while not all(terminals):
      step += 1
      actions = []

      if np.random.random() > epsilon:
        for observation, agent in zip(obs, agents):
          agent.see(observation)
        actions = [agent.action() for agent in agents]
      else:
        actions = [np.random.randint(0, 4) for _ in agents]

      newObs, rewards, terminals = env.step(actions)
      #env.render()
      #time.sleep(0.1)
      # Transform new continous state to new discrete state and count reward
      ep_reward += rewards

      # Every step we update replay memory and train main network
      for ob, agent, action, reward, newOb in zip(obs, agents, actions, rewards, newObs):
        agent.updateReplay(ob, action, reward, newOb, all(terminals))
        agent.train(all(terminals), step)

      obs = newObs
      
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(ep_reward)
    #print("ep_reward: ", ep_reward, " epsilon: ", epsilon, " ep: ", ep)

    # Decay epsilon
    if epsilon > MIN_EPSILON:
      epsilon *= EPSILON_DECAY
      epsilon = max(MIN_EPSILON, epsilon)

  env.close()

  return [rewards.tolist() for rewards in ep_rewards]

def run_multi_agent(env: Env, agents: Sequence[BasicAgent], n_foods, n_eps: int) -> np.ndarray:
  results = np.zeros(n_eps)
  ep_rewards = []

  for ep in range(n_eps):
    step = 0
    terminals = [False for _ in range(len(agents))]
    ep_reward = np.zeros(len(agents))
    obs = env.reset(n_foods)
    env.render()
  
    while not all(terminals):
      step += 1
      actions = []

      for observation, agent in zip(obs, agents):
        agent.see(observation)
      actions = [agent.action() for agent in agents]

      newObs, rewards, terminals = env.step(actions)

      env.render()
      time.sleep(0.1)
      ep_reward += rewards
      obs = newObs
      
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(ep_reward)
    print("ep_reward: ", ep_reward, " ep: ", ep)

  env.close()

  return ep_rewards

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--episodes", type=int, default=2_000)
  parser.add_argument("--image", type=str, default="image")
  parser.add_argument("--agents", type=int, default=1)
  parser.add_argument("--foods", type=int, default=20)
  parser.add_argument("--save", action='store_true')
  parser.add_argument("--load", action='store_true')
  parser.add_argument("--train", action='store_true')
  opt = parser.parse_args()

  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using {DEVICE} |", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

  # 1 - Setup environment
  env = Game(
    gridShape=(10, 10),
    nAgents=opt.agents, nFoods=opt.foods,
    maxSteps=200
  )

  # 2 - Setup agent
  agents = [GameAgent(agentId=id, device = DEVICE) for id in range(0, 1)]
  
  # 3 - Setup agent
  if opt.load:
    for agent in agents:
      agent.load()
  
  # 4 - Evaluate agent
  results = {}
  if opt.train:
    print("training!!!")
    result = train_multi_agent(env, agents, opt.foods, opt.episodes)
  else:
    print("testing!!!")
    result = run_multi_agent(env, agents, opt.foods, opt.episodes)
  results['agent1'] = result

  # 5 - Compare results
  plot(opt.episodes, result, image=opt.image, colors=["orange"])

  # 6 - Save model
  if opt.save:
    for id, agent in enumerate(agents):
      agent.save()

  
  

