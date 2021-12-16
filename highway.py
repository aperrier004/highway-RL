import torch
#from torch._C import ResolutionCallback
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
import random
import highway_env

from models import DQN, Dueling_DQN

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

env = gym.make("highway-fast-v0")
obs_dim = env.observation_space.shape[0]
n_acts = env.action_space.n

### Constants for training
learning_rate = 1e-4
epochs = 300
batch_size = 32
epsilon = 0.1
env.config["duration"] = batch_size
##########################

#############################################
####### BUILDING A NEURAL NETWORK ###########
##### REPRESENTING ACTION STATE VALUES ######
#############################################

# net_Qvalue is a neural network representing an action state value function:
# it takes as inputs observations and outputs values for each action
net_Qvalue = DQN(1, n_acts)

# net_Qvalue_target is another one
net_Qvalue_target = DQN(1, n_acts)
net_Qvalue_target.eval()



def choose_action(observation):
    if random.random() < epsilon:
        return random.randrange(n_acts)
    else:
        with torch.no_grad():
            q_values = net_Qvalue(torch.stack([torch.stack([observation], axis=0)], axis=0))
            return q_values[0].max(0)[1].item()

def compute_loss(batch_observations, batch_actions, batch_rewards, batch_next_observations, batch_non_final):
    batch_q_values = net_Qvalue(torch.stack([batch_observations], axis=1))
    batch_q_value = batch_q_values.gather(0, batch_actions.unsqueeze(1)).squeeze(1)
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
            action = choose_action(torch.as_tensor(observation, dtype=torch.float32))
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

        #if i % 20 == 0:
        print('epoch: %3d \t loss: %.3f' % (i, batch_loss))

DQN()

###### EVALUATION ############

def run_episode(env, render = False):
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        if render:
            env.render()
        action = choose_action(torch.as_tensor(obs, dtype=torch.float32))
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

epsilon = 0
policy_scores = [run_episode(env) for _ in range(100)]
print("Average score of the policy: ", np.mean(policy_scores))

for _ in range(2):
  run_episode(env, True)

env.close()