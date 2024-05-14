import numpy as np
from gym import Env
from typing import Sequence
import argparse
import torch
from utils import plot
from game import Game
import time

from agent.basicAgent import BasicAgent
from agent.greedyDqnAgent import GreedyDQNAgent
import mapGen

EPSILON_DECAY = 0.9998
MIN_EPSILON = 0.005

def train_multi_agent(env: Env, agents: Sequence[BasicAgent], n_foods, n_eps: int, pat: list) -> np.ndarray:
  epsilon = 1
  ep_rewards = []
  gridShape = env.getGridShape()

  for ep in range(n_eps):
    if not (ep % 1000):
      print("ep: ", ep, "epsilon", epsilon)
    step = 0
    terminals = [False for _ in range(len(agents))]
    ep_reward = np.zeros(len(agents))

    obs, info = env.reset(n_foods, pat[np.random.randint(0, len(pat))])
    #env.render()

    for observation, agent in zip(obs, agents):
      agent.see(observation, info)

    if np.random.random() > epsilon:
      spawnActions = [agent.spawnAction() for agent in agents]
      spawnPos = [[act//gridShape[0], act%gridShape[1]] for act in spawnActions]
    else:
      spawnActions = [np.random.randint(0, gridShape[0]*gridShape[1]) for _ in agents]
      spawnPos = [[act//gridShape[0], act%gridShape[1]] for act in spawnActions]
    
    ep_reward += np.array([env.spawn(agent, pos) for agent, pos in zip(agents, spawnPos)])

    newObs = [env.get_agent_obs(agent.id()) for agent in agents]

    for agent in agents:
      agent.resetEnergy()

    while not all(terminals):
      step += 1

      moveActions = {agent.id(): agent.moveAction() for agent in agents}
      _, info, rewards, terminals = env.step(moveActions)
      #env.render()
      #time.sleep(0.1)
      # Transform new continous state to new discrete state and count reward
      ep_reward += rewards

    # Every step we update replay memory and train main network
    for ob, agent, action, reward, newOb in zip(obs, agents, spawnActions, ep_reward, newObs):
      agent.updateReplay(ob, action, reward, newOb, all(terminals))
      agent.train(all(terminals), step)
      
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(np.mean(ep_reward))

    # Decay epsilon
    if epsilon > MIN_EPSILON:
      epsilon *= EPSILON_DECAY
      epsilon = max(MIN_EPSILON, epsilon)

  env.close()

  return [rewards.tolist() for rewards in ep_rewards]

def run_multi_agent(env: Env, agents: Sequence[BasicAgent], n_foods, n_eps: int, pat: list) -> np.ndarray:
  results = np.zeros(n_eps)
  ep_rewards = []
  gridShape = env.getGridShape()

  for ep in range(n_eps):
    step = 0
    terminals = [False for _ in range(len(agents))]
    ep_reward = np.zeros(len(agents))
    obs, info = env.reset(n_foods, pat[np.random.randint(0, len(pat))])
    env.render()

    for observation, agent in zip(obs, agents):
      agent.see(observation, info)
    spawnActions = [agent.spawnAction() for agent in agents]
    spawnPos = [[act//gridShape[0], act%gridShape[1]] for act in spawnActions]

    ep_reward += np.array([env.spawn(agent, pos) for agent, pos in zip(agents, spawnPos)])

    for agent in agents:
      agent.resetEnergy()
      
    while not all(terminals):
      step += 1

      moveActions = {agent.id(): agent.moveAction() for agent in agents}
      newObs, info, rewards, terminals = env.step(moveActions)

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
    gridShape=(50, 50),
    nAgents=opt.agents, nFoods=opt.foods,
    foodCaptureReward=5, maxSteps=15
  )

  # 2 - Setup agent
  agents = [GreedyDQNAgent(agentId=id, nSpawns=50*50, device=DEVICE) for id in range(0, opt.agents)]
  
  # 3 - Setup agent
  if opt.load:
    for id, agent in enumerate(agents):
      agent.load(f"{id}")
  
  pat = [mapGen.map2()]

  # 4 - Evaluate agent
  results = {}
  if opt.train:
    print("training!!!")
    result = train_multi_agent(env, agents, opt.foods, opt.episodes, pat)
  else:
    print("testing!!!")
    result = run_multi_agent(env, agents, opt.foods, opt.episodes, pat)
  results['agent1'] = result

  # 5 - Compare results
  plot(opt.episodes, result, image=opt.image, colors=["orange"])

  # 6 - Save model
  if opt.save:
    for id, agent in enumerate(agents):
      agent.save(f"{id}")
