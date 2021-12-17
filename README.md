# Highway RL
This projects aims to train a DQNAgent on the Gym OpenAI Higway environment.

## Problematic
Keep the car on the highway without having an accident with the cars around. It handles direction, speed and brake.

## Context
Project : Reinforcement Learning
School : ENSC (AI) - Bordeaux INP 2021
Time : 1 week or less

## First steps
The baseline was given by the teacher with the correction of a DQN exercise on the cartpole environnement.
The goal was to make it work on the highway environnement and to improve it.

## Content
- The main file `highway.py` handles the whole thing
- `models.py`contains 2 differents Agents, the basic DQN and the Dueling DQN

## Models
- DQN : The Deep Q Network estimates a Q-value for each action-state pair.
- Dueling DQN : The Dueling network represents two separate estimators: one for the state value function and one for the state-dependent action advantage function.

# Installation
1 - Get the repository

    git clone https://github.com/aperrier004/highway-RL/
    cd highway-RL/


2 - Install it, using Conda for example (use Python >= 3.6)  

    conda create --name myenv python=3.6
    conda activate myenv  
    pip install -r requirements.txt  


3 - Launch a training  

    python highway.py
    
# Results
The trained DQN Dueling manages to change lanes depending on the case to dodge a few cars, but in the best case the score reachs 20 which causes the car to crash after only a few seconds. 

## Intermediate work

Tested activation : tanh, sigmoid, relu, et lin -> stayed with linear because it prevents from normalisation step.

Adaptation of the number of test epochs per training cycle: performance is better with more cycles, the number of epochs per cycle seems less influential. 

Normalization of Q-values: no significant performance increase, not retained.

## Additional ideas 

Work with a memory buffer. Adjust the hyper-parameters of the model. 
