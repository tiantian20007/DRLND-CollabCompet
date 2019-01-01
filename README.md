# DRLND-CollabCompet
Collaboration and Competition project for Deep Reinforcement Learning Nanodegree Program

# Environment
For this project, we will work with the Tennis environment provided by Udacity.

![alt text][env]

[env]: https://github.com/tiantian20007/DRLND-CollabCompet/blob/master/res/env.png "Tennis"

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

__Every entry in the action vector should be a number between -1 and 1.__

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

# Environment setup

## Step 1: Clone the DRLND Repository
If you haven't already, please follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

## Step 2: Download the Unity Environment - Tennis 
For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

**Then, place the file in the root folder in the this cloned repository, and unzip (or decompress) the file.**

(For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

# Run the code
Type 'python main.py' in terminal and press Enter key.

The trained parameters will be save as:
- checkpoint_actor_0.pth
- checkpoint_actor_1.pth
- checkpoint_critic_0.pth
- checkpoint_critic_1.pth

We can use 'watch_smart_agent' function to load that parameters to our network and watch the traned agent play.
There's also a 'play' function to watch random agent play.

# Result
Our agent got an average score of +0.5 over 100 consecutive episodes after 963 episodes traning.

![alt text][result]

[result]: https://github.com/tiantian20007/DRLND-CollabCompet/blob/master/res/obs_full.png "Result"
