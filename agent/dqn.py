import numpy as np
import torch
import torch.nn as nn

from collections import deque
import random
import os

MIN_REPLAY_MEMORY_SIZE = 512
REPLAY_MEMORY_SIZE = 16_384 # 2^14
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 4
DISCOUNT = 0.95

class MLP(nn.Module):
  '''
    @hiddenLayer : an list of hidden layer sizes.
    @outputLayer : the output layer size.
  '''
  def __init__(self, outputLayer, dropout=0.1):
    super(MLP, self).__init__()

    self.model_ = nn.Sequential(
      # shape = [Batch_size, 3, 10, 10]
      # formula = (10 - kernel + 2*padding)/stride + 1
      nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(4, 4), stride=2),
      nn.ReLU(), # shape = [Batch_size, 16, 4, 4]
      nn.Dropout(p=dropout),

      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2), stride=1),
      nn.ReLU(), # shape = [Batch_size, 32, 3, 3]
      nn.Dropout(p=dropout),
        
      nn.Flatten(),

      nn.Linear(32*3*3, 64),
      nn.ReLU(),
      nn.Dropout(p=dropout),

      nn.Linear(64, outputLayer)
    )

  def forward(self, x: torch.Tensor):
    #x = torch.tensor(np.array(x), dtype=torch.float32).to(self.device_)
    #print("self.model_(x) ", self.model_(x))
    return self.model_(x)

class DQN():

  def __init__(self, outputLayer, dropout=0.1, device: str = "cpu"):
    
    self.device_ = torch.device(device)

    self.model_ = MLP(outputLayer, dropout).to(device)
    self.targetModel_ = MLP(outputLayer, dropout).to(device)
    self.targetModel_.load_state_dict(self.model_.state_dict())

    # An array with last n steps for training
    self.replayMemory_ = deque(maxlen=REPLAY_MEMORY_SIZE)
    self.replayMemorySize_ = 0

    # Used to count when to update target network with main network's weights
    self.targetUpdateCounter_ = 0

    self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=0.01)
    
    #self.criterion_ = nn.SmoothL1Loss()
    self.criterion_ = nn.MSELoss()

  # Return the Q-values.
  @torch.no_grad()
  def getQs(self, state):
    state = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device_)
    return self.targetModel_(state).cpu().detach().numpy().squeeze()
  
  def updateReplay(self, obs, action, reward, newObs, done):
    self.replayMemorySize_ += 1
    self.replayMemory_.append((obs, action, reward, newObs, done))

  def train(self, terminal_state, step):

    # Start training only if certain number of samples is already saved
    if self.replayMemorySize_ < MIN_REPLAY_MEMORY_SIZE:
      return
    
    # Get a minibatch of random samples from memory replay table
    minibatch = random.sample(self.replayMemory_, MINIBATCH_SIZE)

    current_states = torch.FloatTensor(np.array([transition[0] for transition in minibatch])/255).to(self.device_)
    rewards = torch.FloatTensor(np.array([transition[2] for transition in minibatch])).to(self.device_)
    new_current_states = torch.FloatTensor(np.array([transition[3] for transition in minibatch])/255).to(self.device_)
    dones = torch.FloatTensor(np.array([transition[4] for transition in minibatch])).to(self.device_)

    future_qs_list = self.targetModel_(new_current_states)
    y = self.model_(current_states).max(1).values
    yHat = rewards + DISCOUNT * future_qs_list.max(dim=1)[0] * (1 - dones)

    y = y.to(self.device_)
    self.optimizer_.zero_grad()
    loss = self.criterion_(y, yHat)
    loss.backward()
    self.optimizer_.step()
    
    # Update target network counter every episode
    if terminal_state:
        self.targetUpdateCounter_ += 1

    # If counter reaches set value, update target network with weights of main network
    if self.targetUpdateCounter_ > UPDATE_TARGET_EVERY:
      self.targetModel_.load_state_dict(self.model_.state_dict())
      self.targetUpdateCounter_ = 0

    return loss.item()
  
  def save(self, file_name='model.pth'):
    model_folder_path = './model'
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
    
    file_name = os.path.join(model_folder_path, file_name)
    torch.save({
                'model_state_dict': self.model_.state_dict(),
                'optimizer_state_dict': self.optimizer_.state_dict(),
                }, file_name)
    
    for param_tensor in self.model_.state_dict():
      print(param_tensor, "\t", self.model_.state_dict()[param_tensor].size())

  def load(self, file_name='model.pth'):
    model_folder_path = './model'
    if not os.path.exists(model_folder_path):
      print(" > No model found, initializing a default one")
      return
    file_name = os.path.join(model_folder_path, file_name)

    checkpoint = torch.load(file_name)
    self.model_.load_state_dict(checkpoint['model_state_dict'])
    self.targetModel_.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer_.load_state_dict(checkpoint['optimizer_state_dict'])

    self.model_.eval()
    self.targetModel_.eval()
    

