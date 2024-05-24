import numpy as np
from gym import Env
from typing import Sequence
import argparse
import torch
from utils import plot, plot3d
from game import Game
import time

from agent.gameAgentFactory import GameAgentFactory
from agent.dqnAgent import DQNAgent
import mapGen

SPAWN_PENALTY = -15

def train_multi_agent(env: Env, agents: Sequence[DQNAgent], n_foods, n_eps: int, pat: list) -> tuple:
  ep_rewards = []
  ep_strengths = [[agent.getStrength() for agent in agents]]
  ep_speeds = [[agent.getSpeed() for agent in agents]]
  gridShape = env.getGridShape()

  for ep in range(1, n_eps+1):
    if (not (ep % 10)) and ep: # Update every 10 ep
      means = np.mean(np.array(ep_rewards[-10:]), axis=0)
      for i, agent in enumerate(agents):
        agent.updateGenetic(reward=means[i])
      print(f"ep: {ep}/{n_eps} | DQN epsilon: {agents[0].epsilonSpawn()} | QL epsilon: {agents[0].epsilonGenetic()}", end="\r")
        
      ep_strengths.append([agent.getStrength() for agent in agents])
      ep_speeds.append([agent.getSpeed() for agent in agents])

    step = 0
    terminals = [False for _ in range(len(agents))]
    ep_reward = np.array([SPAWN_PENALTY for _ in range(len(agents))], dtype=np.float32)

    obs, info = env.reset(n_foods, pat[np.random.randint(0, len(pat))])

    for observation, agent in zip(obs, agents):
      agent.see(observation, info)

    spawnActions = [agent.spawnAction() for agent in agents]
    spawnPos = [[act//gridShape[0], act%gridShape[1]] for act in spawnActions]
    
    ep_reward += np.array([env.spawn(agent, pos) for agent, pos in zip(agents, spawnPos)])

    newObs = [env.get_agent_obs(agent.getId()) for agent in agents]

    for agent in agents:
      agent.resetEnergy()

    while not all(terminals):
      step += 1

      agentSteps = [agent.moveSteps() for agent in agents]
      while not all(x == 0 for x in agentSteps):
        for i in range(0, len(agentSteps)):
          if (agentSteps[i]):
            agentSteps[i] -= 1
        moveActions = {agent.getId(): agent.moveAction() for agent in agents}
        _, info, _, terminals = env.step(moveActions)

      for agent in agents:
        agent.moved() # notify that the agent moved, so it will decrease energy

      rewards = env.doColisions()

      # Transform new continous state to new discrete state and count reward
      ep_reward += rewards

    # Every step we update replay memory and train main network
    for ob, agent, action, reward, newOb in zip(obs, agents, spawnActions, ep_reward, newObs):
      agent.updateSpawnReplay(ob, action, reward, newOb, all(terminals))
      agent.train(all(terminals), step)
      
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(ep_reward)

  env.close()

  return (np.mean(np.array(ep_rewards), axis=1).tolist(), 
          np.mean(np.array(ep_strengths), axis=1).tolist(), 
          np.mean(np.array(ep_speeds), axis=1).tolist())

def run_multi_agent(env: Env, agents: Sequence[DQNAgent], n_foods, n_eps: int, pat: list) -> np.ndarray:
  ep_rewards = []
  ep_strengths = [[agent.getStrength() for agent in agents]]
  ep_speeds = [[agent.getSpeed() for agent in agents]]
  gridShape = env.getGridShape()

  for ep in range(1, n_eps+1):
    if (not (ep % 10)) and ep: # Update every 10 ep
      means = np.mean(np.array(ep_rewards[-10:]), axis=0)
      for i, agent in enumerate(agents):
        agent.updateGenetic(reward=means[i])
      print(f"ep: {ep}/{n_eps} | DQN epsilon: {agents[0].epsilonSpawn()} | QL epsilon: {agents[0].epsilonGenetic()}", end="\r")
        
      ep_strengths.append([agent.getStrength() for agent in agents])
      ep_speeds.append([agent.getSpeed() for agent in agents])


    step = 0
    terminals = [False for _ in range(len(agents))]
    ep_reward = np.array([SPAWN_PENALTY for _ in range(len(agents))], dtype=np.float32)
    obs, info = env.reset(n_foods, pat[np.random.randint(0, len(pat))])
    
    if (ep > 4000): 
      env.render()
      time.sleep(0.5)

    for observation, agent in zip(obs, agents):
      agent.see(observation, info)
    spawnActions = [agent.spawnAction() for agent in agents]
    spawnPos = [[act//gridShape[0], act%gridShape[1]] for act in spawnActions]

    ep_reward += np.array([env.spawn(agent, pos) for agent, pos in zip(agents, spawnPos)])
    if (ep > 4000): 
      env.render() 
      time.sleep(0.2)

    for agent in agents:
      agent.resetEnergy()
      
    while not all(terminals):
      step += 1
      
      agentSteps = [agent.moveSteps() for agent in agents]
      print(agentSteps)
      while not all(x == 0 for x in agentSteps):
        for i in range(0, len(agentSteps)):
          if (agentSteps[i]):
            agentSteps[i] -= 1
        moveActions = {agent.getId(): agent.moveAction() for agent in agents}
        _, info, _, terminals = env.step(moveActions)
        if (ep > 4000): 
          env.render()
          time.sleep(0.1)

      for agent in agents:
        agent.moved() # notify that the agent moved, so it will decrease energy

      rewards = env.doColisions()
      
      if (ep > 4000): 
        env.render()
        time.sleep(0.1)
      ep_reward += rewards
      
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(ep_reward)
    #print("ep_reward: ", ep_reward, " ep: ", ep)

  env.close()

  return (np.mean(np.array(ep_rewards), axis=1).tolist(), 
          np.mean(np.array(ep_strengths), axis=1).tolist(), 
          np.mean(np.array(ep_speeds), axis=1).tolist())

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
  seed = None if opt.train else 0
  print("Set seed to: ", seed)
  env = Game(
    gridShape=(30, 30), 
    nFoods=opt.foods,
    foodCaptureReward=5, seed=seed
  )

  # 2 - Setup agent
  factory = GameAgentFactory(seed=seed)
  agents = [factory.createGreedyDqnAgent(maxEnergy=100, nSpawns=30*30, device=DEVICE, train=opt.train) 
            for _ in range(0, opt.agents)]
  
  for agent in agents:
    env.addAgent(agent.getId(), agent)

  # 3 - Setup agent
  if opt.load:
    for agent in agents:
      agent.load(prefix=f"{agent.getId()}")
  
  pat = [mapGen.map4Corner(30, 30)]

  # 4 - Evaluate agent
  results = {}
  if opt.train:
    print("training!!!")
    results['rewards'], results['mean_strength'], results['mean_speed'] = train_multi_agent(env, agents, opt.foods, opt.episodes, pat)
  else:
    print("testing!!!")
    results['rewards'], results['mean_strength'], results['mean_speed'] = run_multi_agent(env, agents, opt.foods, opt.episodes, pat)

  # 5 - Compare results
  plot(y=results['rewards'], x=np.arange(0, len(results['rewards'])),
       xLabel = 'Episodes', yLabel = 'Scores',
       ylim=(-15, 15),
       s=0.1, image=f"images/results/{opt.image}Rewards", colors=["orange"])
  
  plot(y=results['mean_strength'], x=np.arange(0, len(results['mean_strength'])),
       xLabel = 'Updates', yLabel = 'Strength',
       ylim=(0, 10),
       s=3, image=f"images/results/{opt.image}Strength", colors=["orange"])
  
  plot(y=results['mean_speed'], x=np.arange(0, len(results['mean_speed'])),
       xLabel = 'Updates', yLabel = 'Speed',
       ylim=(0, 10),
       s=3, image=f"images/results/{opt.image}Speed", colors=["orange"])
  
  final_strengths = [agent.getStrength() for agent in agents]
  final_speeds = [agent.getSpeed() for agent in agents]
  ids = [agent.getId()+1 for agent in agents]

  plot3d(x=final_speeds, y=final_strengths, z=ids,
      xLabel = 'Speed', yLabel = 'Strength', zLabel = 'AgentId',
      xlim=(0, 10), ylim=(0, 10),
      s=3, image=f"images/results/{opt.image}Final", colors=["blue"])
  
  # 6 - Save model
  if opt.save:
    for agent in agents:
      agent.save(prefix=f"{agent.getId()}")
