import numpy as np
from gym import Env
from typing import Sequence
import argparse
import torch
from utils import plot
from game import Game
import time

from agent.gameAgentFactory import GameAgentFactory
from agent.dqnAgent import DQNAgent
import mapGen

EPSILON_DECAY = 0.9998
MIN_EPSILON = 0.005

SPAWN_PENALTY = -15

def train_multi_agent(env: Env, agents: Sequence[DQNAgent], n_foods, n_eps: int, pat: list) -> tuple:
  epsilon = 1
  ep_rewards = []
  ep_strengths = [np.mean(np.array([agent.getStrength() for agent in agents], dtype=np.float32))]
  gridShape = env.getGridShape()

  for ep in range(1, n_eps+1):
    if (not (ep % 100)) and ep: # Update every 100 ep
      print("ep: ", ep, "epsilon", epsilon)
      for agent in agents:
        agent.updateGenetic(np.mean(np.array(ep_rewards[-100:])))
      ep_strengths.append(np.mean(np.array([agent.getStrength() for agent in agents], dtype=np.float32)))

    step = 0
    terminals = [False for _ in range(len(agents))]
    ep_reward = np.array([SPAWN_PENALTY for _ in range(len(agents))], dtype=np.float32)

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

    newObs = [env.get_agent_obs(agent.getId()) for agent in agents]

    for agent in agents:
      agent.resetEnergy()

    while not all(terminals):
      step += 1

      moveActions = {agent.getId(): agent.moveAction() for agent in agents}
      _, info, rewards, terminals = env.step(moveActions)
      #env.render()
      #time.sleep(0.1)
      # Transform new continous state to new discrete state and count reward
      ep_reward += rewards

    # Every step we update replay memory and train main network
    for ob, agent, action, reward, newOb in zip(obs, agents, spawnActions, ep_reward, newObs):
      agent.updateSpawnReplay(ob, action, reward, newOb, all(terminals))
      agent.train(all(terminals), step)
      
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(np.mean(ep_reward))

    # Decay epsilon
    if epsilon > MIN_EPSILON:
      epsilon *= EPSILON_DECAY
      epsilon = max(MIN_EPSILON, epsilon)

  env.close()

  #return ([rewards.tolist() for rewards in ep_rewards], ep_strengths)
  return (ep_rewards, ep_strengths)

def run_multi_agent(env: Env, agents: Sequence[DQNAgent], n_foods, n_eps: int, pat: list) -> np.ndarray:
  ep_rewards = []
  ep_strengths = [np.mean(np.array([agent.getStrength() for agent in agents], dtype=np.float32))]
  gridShape = env.getGridShape()

  for ep in range(1, n_eps+1):
    if (not (ep % 100)) and ep: # Update every 100 ep
      for agent in agents:
        agent.updateGenetic(np.mean(np.array(ep_rewards[-100:])))
      ep_strengths.append(np.mean(np.array([agent.getStrength() for agent in agents], dtype=np.float32)))

    step = 0
    terminals = [False for _ in range(len(agents))]
    ep_reward = np.array([SPAWN_PENALTY for _ in range(len(agents))], dtype=np.float32)
    obs, info = env.reset(n_foods, pat[np.random.randint(0, len(pat))])
    #env.render()

    for observation, agent in zip(obs, agents):
      agent.see(observation, info)
    spawnActions = [agent.spawnAction() for agent in agents]
    spawnPos = [[act//gridShape[0], act%gridShape[1]] for act in spawnActions]

    ep_reward += np.array([env.spawn(agent, pos) for agent, pos in zip(agents, spawnPos)])

    for agent in agents:
      agent.resetEnergy()
      
    while not all(terminals):
      step += 1

      moveActions = {agent.getId(): agent.moveAction() for agent in agents}
      newObs, info, rewards, terminals = env.step(moveActions)

      #env.render()
      #time.sleep(0.1)
      ep_reward += rewards
      obs = newObs
      
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(np.mean(ep_reward))
    print("ep_reward: ", ep_reward, " ep: ", ep)

  env.close()

  return (ep_rewards, ep_strengths)

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
    gridShape=(30, 30), 
    nFoods=opt.foods,
    foodCaptureReward=5, maxSteps=10
  )

  # 2 - Setup agent
  factory = GameAgentFactory(seed=0)
  agents = [factory.createGreedyDqnAgent(maxEnergy=20, 
                                         nSpawns=30*30, nGenetics=2, device=DEVICE) 
            for _ in range(0, opt.agents)]
  
  for agent in agents:
    env.addAgent(agent.getId(), agent)

  # 3 - Setup agent
  if opt.load:
    for agent in agents:
      agent.load(prefix=f"{agent.getId()}")
  
  pat = [mapGen.map2(30, 30)]

  # 4 - Evaluate agent
  results = {}
  if opt.train:
    print("training!!!")
    results['rewards'], results['strength'] = train_multi_agent(env, agents, opt.foods, opt.episodes, pat)
  else:
    print("testing!!!")
    results['rewards'], results['strength'] = run_multi_agent(env, agents, opt.foods, opt.episodes, pat)

  # 5 - Compare results
  plot(xLen=opt.episodes, x=results['rewards'], 
       xLabel = 'Episodes', yLabel = 'Scores',
       ylim=(-20, 15), s=0.1, 
       image=f"{opt.image}rewards", colors=["orange"])
  
  plot(xLen=opt.episodes//100+1, x=results['strength'], 
       xLabel = 'Updates', yLabel = 'Strength',
       ylim=(-2, 2), s=2, 
       image=f"{opt.image}strength", colors=["orange"])

  # 6 - Save model
  if opt.save:
    for agent in agents:
      agent.save(prefix=f"{agent.getId()}")
