# Method

## Algorithm

MADDPG - Multi-Agent Deep Deterministic Policy Gradient Algorithm

![alt text](https://github.com/tiantian20007/DRLND-CollabCompet/blob/master/res/algorithm2.png "algorithm")

## Model architectures

A DNN with 2 hidden layers which contains 256 and 128 neuron units for both actor and critic. We choose relu as the activate function. 
First, I tried critics for each agent with full observation of all agent as input, then an modified critics with only local observation of each agent, the two models all generate the qulified result.

The one with full observation train a little bit long with 963 episode and the other one only took around 800 episode.
I think the environment is not very complex, so the modified simpler version still qualify. 

![alt text](https://github.com/tiantian20007/DRLND-CollabCompet/blob/master/res/algorithm.png "algorithm")


## Hyperparameters

- BUFFER_SIZE = int(1e6)  # replay buffer size
- BATCH_SIZE = 512        # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 0.1               # for soft update of target parameters
- LR_ACTOR = 1e-4         # learning rate of the actor 
- LR_CRITIC = 1e-3        # learning rate of the critic
- WEIGHT_DECAY = 0        # L2 weight decay

- noise_start=1.0
- noise_reduction=0.99995

# Rewards

Critic with full observation of all agents.

![alt text](https://github.com/tiantian20007/DRLND-CollabCompet/blob/master/res/obs_full.png "obs_full")

Critic with only local observation of each agent.

![alt text](https://github.com/tiantian20007/DRLND-CollabCompet/blob/master/res/obs.png "obs")


# Future Work

## [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

Our algorithm samples experience transitions uniformly from a replay memory. 
Prioritized experienced replay is based on the idea that the agent can learn more effectively from some transitions than from others, and the more important transitions should be sampled with higher probability.

![alt text](https://github.com/tiantian20007/DRLND-Navigation/blob/master/res/Prioritized-Experience-Replay.png "Prioritized Experience Replay")

## Agents with Policy Ensembles

I got the idea from the paper: [Multi Agent Actor Critic for Mixed Cooperative Competitive environments](https://arxiv.org/abs/1706.02275) 
we can a recurring problem in multi-agent reinforcement learning is the environment non-stationarity due to the agents’ changing policies. This is particularly true in competitive settings, where agents can derive a strong policy by overfitting to the behavior of their competitors.
Such policies are undesirable as they are brittle and may fail when the competitors alter strategies.To obtain multi-agent policies that are more robust to changes in the policy of competing agents,
we propose to train a collection of K different sub-policies. At each episode, we randomly select one particular sub-policy for each agent to execute.

## PPO
I got the idea from the paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
It's a modified versoin of policy gradient method. I think we can improve the actor part of the MADDPG algorithm.

![alt text](https://github.com/tiantian20007/DRLND-CollabCompet/blob/master/res/ppo.png "Result")
