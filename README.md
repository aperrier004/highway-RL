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

## Results
The results are pretty bad (the mean score is around 20) with the DQN and also the dueling DQN.

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
