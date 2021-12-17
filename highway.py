# Author : Alban PERRIER (https://github.com/aperrier004/highway-RL)
# Context : RL project at Bordeaux INP (ENSC - IA)
# Goal : Implementation of DQN on the highway env (https://github.com/eleurent/highway-env)
# Baseline : DQN exercice of Nathanael Fijalkow (https://github.com/nathanael-fijalkow/nathanael-fijalkow.github.io)

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
import math, random
import highway_env

# Import classes of models.py
from models import DQN, Dueling_DQN

# dll error, not mandatory on every environnements
# https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

### Highway fast Environnement
env = gym.make("highway-fast-v0")
obs_dim = env.observation_space.shape[0]
n_acts = env.action_space.n

### Epsilon greedy exploration
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

# Computes the epsilon for each episodes
epsilon_by_epoch = lambda i: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * i / epsilon_decay)

# Modify this to change the type of DQN
model = "Dueling_DQN"

###### Deep Q Network ############
# net_Qvalue is a neural network representing an action state value function:
# it takes as inputs observations and outputs values for each action
net_Qvalue = DQN(1, n_acts)

# net_Qvalue_target is another one
net_Qvalue_target = DQN(1, n_acts)

if model == "Dueling_DQN":
    ### Dueling Deep Q Network
    net_Qvalue = Dueling_DQN(1, n_acts)
    net_Qvalue_target = Dueling_DQN(1, n_acts)

net_Qvalue_target.eval()

###### TRAINING ############
### Constants for training
learning_rate = 0.00001
epochs = 100
batch_size = 32
#epsilon = 0.1 # Replaced by epsilon greedy exploration


def choose_action(observation, epsilon):
    """Choose an action for an episode
    
    Args:
        observation (float32 Tensor): the current state of the environnement with its conditions
        
    Returns:
        q_values (int): value of the action to realize (either random or the maximum value of q_values[0])
    """
    if random.random() < epsilon:
        return random.randrange(n_acts)
    else:
        with torch.no_grad():
            # Concatenates observation tensors along  axis 0
            q_values = net_Qvalue(torch.stack([torch.stack([observation], axis=0)], axis=0))
            return q_values[0].max(0)[1].item()

def compute_loss(batch_observations, batch_actions, batch_rewards, batch_next_observations, batch_non_final):
    """Compute the loss for an epoch
    
    Args:
        batch_observations (float32 Tensor): list of all the observations
        batch_actions (float32 Tensor): list of all the actions
        batch_rewards (float32 Tensor): list of all the reward
        batch_next_observations (float32 Tensor): list of all the next observations
        batch_non_final (float32 Tensor): list that counts the episodes not done
        
    Returns:
        loss (int): value of the loss for the neural network
    """
    # Current model with batch_observations tensors concatenated along axis 1 
    batch_q_values = net_Qvalue(torch.stack([batch_observations], axis=1))
    # Target model
    # Gathers values along the 0 axis 
    batch_q_value = batch_q_values.gather(0, batch_actions.unsqueeze(1)).squeeze(1) # tensor with a dimension of size one inserted at the position 1 and then removed
    #print("q_values",batch_q_values.shape)
    #print("actions", len(batch_actions))

    batch_q_value_next = torch.zeros_like(batch_q_value)
    with torch.no_grad():
        next_non_final_observations = batch_next_observations[batch_non_final]
        batch_q_values_next = net_Qvalue(torch.stack([next_non_final_observations], axis=1))
        _, batch_max_indices = batch_q_values_next.max(dim=1)
        batch_q_values_next = net_Qvalue_target(torch.stack([next_non_final_observations], axis=1))
        batch_q_value_next[batch_non_final] = batch_q_values_next.gather(1, batch_max_indices.unsqueeze(1)).squeeze(1)

    batch_expected_q_value = batch_rewards + batch_q_value_next
    loss = (batch_q_value - batch_expected_q_value).pow(2).mean()

    return loss

# make optimizer
optimizer = Adam(net_Qvalue.parameters(), lr = learning_rate)

def DQN():
    """Training the Deep Q Network"""
    for i in range(epochs):
        # we copy the parameters of Qvalue into Qvalue_target every 10 iterations
        if i % 10 == 0:
            net_Qvalue_target.load_state_dict(net_Qvalue.state_dict())

        batch_observations = [] 
        batch_actions = []      
        batch_rewards = []      
        batch_next_observations = [] 
        batch_non_final = []

        # for statistics over all episodes run in the first step
        episodes = 0
        total_reward = 0

        # reset episode-specific variables
        observation = env.reset()
        done = False

        # First step: collect experience by simulating the environment using the current policy
        while True:
            old_observation = observation.copy()
            epsilon = epsilon_by_epoch(i)
            action = choose_action(torch.as_tensor(observation, dtype=torch.float32), epsilon)
            observation, reward, done, _ = env.step(action)
            total_reward += reward

            batch_observations.append(old_observation)
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_next_observations.append(observation)
            batch_non_final.append(not done)

            # end the While loop if we have enough experiments
            if len(batch_observations) > batch_size:
                if i % 20 == 0:
                    print(f"Mean episode reward:{total_reward / episodes:.2f}")
                    if total_reward / episodes >= 250:
                        return
                break

            if done:
                # reset episode-specific variables
                observation, done = env.reset(), False
                episodes += 1

        # Second step: update the policy
        # we take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(torch.as_tensor(np.array(batch_observations), dtype=torch.float32),
                                  torch.as_tensor(batch_actions, dtype=torch.int64),
                                  torch.as_tensor(batch_rewards, dtype=torch.float32),
                                  torch.as_tensor(np.array(batch_next_observations), dtype=torch.float32),
                                  torch.as_tensor(batch_non_final, dtype=torch.bool)
                                  )
        batch_loss.backward()
        optimizer.step()

        print('epoch: %3d \t loss: %.3f' % (i, batch_loss))

DQN()

###### EVALUATION ############

def run_episode(env, i, render = False):
    """Testing cycle"""
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        if render:
            env.render()
            
        epsilon = epsilon_by_epoch(i)
        action = choose_action(torch.as_tensor(obs, dtype=torch.float32), epsilon)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

epsilon = 0
policy_scores = [run_episode(env, i) for i in range(100)]
print("Average score of the policy: ", np.mean(policy_scores))

for i in range(10):
  run_episode(env, i, True)

env.close()