import numpy as np
import os
import pickle

'''
  Adapted based on: 
    https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial1/qlearn.py
'''

INC_STRENGTH, DEC_STRENGTH, INC_SPEED, DEC_SPEED, NO_OP = range(5)

class QLearning():

  def __init__(self, nActions, alpha=0.2, gamma=0.8, seed = None):
    self.qTable_ = {} # Q-table

    self.alpha_ = alpha           # Discount constant
    self.gamma_ = gamma           # Discount factor
    self.nActions_ = nActions     # Number of actions

    self.seeds_ = seed            # Seed
    np.random.seed(seed)

  def seed(self): return self.seeds_

  def alpha(self): return self.alpha_
  def gamma(self): return self.gamma_

  # Return the Q-values.
  def getQ(self, state, action):
    """
    Get Q value for a state-action pair.

    If the state-action pair is not found in the dictionary,
        return 0.0 if not found in our dictionary
    """
    return self.qTable_.get((state, action), 0.0)

  def learnQ(self, state, action, reward, newState):
    '''
    Q-learning:
        Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s')) - Q(s,a))            
    '''        
    qValue = self.qTable_.get((state, action), None)
    if qValue is None:
      self.qTable_[(state, action)] = reward
    else:
      newQValue = max([self.getQ(newState, a) for a in range(0, self.nActions_)])
      self.qTable_[(state, action)] = qValue + self.alpha_ * (reward + self.gamma_*newQValue - qValue)
  
  def chooseAction(self, state):
    """
    Epsilon-Greedy approach for action selection.
    """      
    return np.argmax(np.array([self.getQ(state, a) for a in range(0, self.nActions_)]))
  
  def save(self, file_name='QLmodel'):
    model_folder_path = 'model'
    file_name = os.path.join(model_folder_path, file_name)
    
    with open(file_name, 'wb') as f:
      pickle.dump(self.qTable_, f)

  def load(self, file_name='QLmodel'):
    model_folder_path = 'model'
    file_name = os.path.join(model_folder_path, file_name)

    with open(file_name, 'rb') as f:
      self.qTable_ = pickle.load(f)