
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"
[image3]: plot.png "plot"
# Deep Reinforcement Learning - DDPG Algorithm  Reacher Continuous Control 
![Trained Agent][image1]


### Introduction

For this project, I worked with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

####  Solve the Second Version
The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, My agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 
To solve the challenge i used the architecture:
### Model Architecture
The Udacity provided DDPG code in PyTorch was used and adapted for this 20 agent (version 2) environment.

The algorithm uses two deep neural networks (actor-critic) with the following struture:
- Actor    
	- Batch Normalisation for the input
    - Hidden: Fully connected : (input, 128)  -> ReLU
    - Hidden: Fully connected : (128, 128)    -> ReLU
    - Ouptut: Fully connected : (128, 4)       -> TanH

- Critic
 	- Batch Normalisation for the input
    - Hidden: Fully connected : (input, 128)  -> ReLU
    - Hidden: Fully connected : (128 + action size , 128)    -> ReLU
    - Output : Fully connected : (128, 1)       

### Hyperparameters
- Learning Rate: 1e-4 (in both DNN)
- Batch Size: 256
- Replay Buffer: 1e5
- Gamma: 0.99
- Tau: 1e-3

### Code implementation
My code consist of :
#### model.py : 
 two deep neural networks (actor-critic) with 20 agents and the output action are 4, the 4 movement.
#### ddpg_agent.py : 
A  DDPG code in PyTorch was used and adapted for this 20 agent (version 2) environment.
#### Continuous_Control.ipynb :
Jupyter notebooks of the main project, used to train the model and plot and test the results. Steps:
-   Import resources Packages
-   Test the State and Action Spaces
-   Test Random Actions
-   Train an agent using DQN
-   Plot the scores



# RESULTS:

![Result][image3]


### TODO : Challenge: Crawler Environment

It is a more difficult **Crawler** environment.

![Crawler][image2]

In this continuous control environment, the goal is to teach a creature with four legs to walk forward without falling.  

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler).  To solve this harder task, you'll need to download a new Unity environment.  (**Note**: Udacity students should not submit a project with this new environment.)








