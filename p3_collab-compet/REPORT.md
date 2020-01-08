
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"
[plot]: plot.png "plot"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

###  Algorithms and Techniques

To reach to goal i used MDDGP algorithm.
MADDPG (Multi-agent DDPG) class uses 2 DDPG agents (Actor and Critic), it is similar to the model used in Udacity classroom. 

The Actor network consists of:
- Linear layers of 24*200 followed by Relu layer 
- Linear layers of 200*150 followed by Relu layer 
- Linear layer of 150*2 is followed by a Tanh layer used as output layer. 

The Critic network consist of:
- Linear layers of 52*200 followed by Relu layer 
- Linear layers of 200*150  followed by Relu layer 
- Linear layer of 150*1 is followed by a Tanh layer used as output layer. 

Each agent receives its own environment separately. So both agents train simultaneously through self-play. Also ReplayBuffer is used as a shared buffer between agents.

 It is from the paper  [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275), and i started from the hyper parameter mentioned in the paper
 
 ### DDPG Agent Hyper Parameters

-   BUFFER_SIZE (int): replay buffer size `BUFFER_SIZE = int(1e5)`
-   BATCH_SIZE (int): mini batch size `BATCH_SIZE = 256` then by test i used `240`
-   GAMMA (float): discount factor `GAMMA = 0.99`
-   TAU (float): for soft update of target parameters `TAU = 1e-3`
-   LR_ACTOR (float): learning rate for optimizer `LR_ACTOR = 1e-3` then i changed it to ` 1e-4`
-   LR_CRITIC (float): learning rate for optimizer `LR_CRITIC = 1e-3` then i changed it to ` 1e-4`
-   WEIGHT_DECAY (float): L2 weight decay `WEIGHT_DECAY = 0.0`

### Results

We reached the aimed result after 2900 episodes, with average score:  0.5 for last 100 episodes. 
The weights of two agents are saved in :
- agent1_checkpoint_actor.pth
- agent1_checkpoint_critic.pth
- agent2_checkpoint_actor.pth
- agent2_checkpoint_critic.pth

environment was usually around 2000.
#### Learning graph
Graphs for episodes scores and average scores are as follows:
![Plot][plot]


### Future Plans for Improvement

I consider continue working on the project in the following axes:
-   Optimise Hyperparameter 
Many of the hyperparameters, such as the network architectures (number of layers and neurons), learning rates, batch and buffer sizes, and the level of noise, are took by tests and they could be further tuned to improve performance.
    
-   Alternative learning algorithms
Alternative methods algorithm can be tested, example include PPO with team spirit and the multi-agent DDPG algorithm described that uses centralised training and decentralised execution.
