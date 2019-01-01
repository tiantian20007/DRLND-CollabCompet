# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 20:06:26 2018

@author: tianTian
"""

from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from itertools import count
import time

from buffer import ReplayBuffer
from maddpg import MADDPG
from utilities import transpose_list, transpose_to_tensor


batchsize = 512
BUFFER_SIZE = int(1e6)  # replay buffer size

env = 0
brain_name = 0
brain = 0
state_size = 0
action_size = 0
num_agents = 0


def loadEnv():
    global env
    global brain_name
    global brain
    global state_size
    global action_size
    global num_agents
    env = UnityEnvironment(file_name="./Tennis_Windows_x86_64/Tennis.exe")
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    
    # number of agents 
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    
    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)
    
    # examine the state space 
    states = env_info.vector_observations
    #print(states)
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])
    print('The state for the second agent looks like:', states[1])
    
def closeEnv():
    global env
    env.close()
    

def play(random=True, n_episode=5, agent=None):
    global env
    global brain_name
    global brain
    global state_size
    global action_size
    global num_agents
    
    for i in range(n_episode):                                      # play game for 5 episodes
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        while True:
#        for t in range(1000):
            if random == True:
                actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
                actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            else:
                actions = agent.act(states, noise=0)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
        print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))    
    
    
def maddpgTrain(n_episodes=1000):
    global env
    global brain_name
    global brain
    global state_size
    global action_size
    global num_agents
    
    # one buffer for a agent
    buffer = ReplayBuffer(BUFFER_SIZE, num_agents)
    
    # init agent
    agent = MADDPG(in_actor=state_size, out_actor=action_size, agent_count=num_agents)
    
    scores_deque = deque(maxlen=100)
    scores_global=[]
    scores_average=[]
    max_score = -np.Inf
    totalTimeTaken=0
    t = 0

    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 1
    noise_reduction = 0.99995
    n_episode_random = 300
    n_max_t = 1000
    learn_rate = 1
    learn_count = 3
    
    for i_episode in range(n_episodes):  
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)        
        scoresPerEpisode = np.zeros(num_agents)                # initialize the score (for each agent)
        
        sTime = time.time()
        for t in range(n_max_t):
#        for t in count():
            # action input needs to be transposed
            actions = agent.act(states, noise=noise)
            
            env_info= env.step(actions)[brain_name] ## env goes to next stage
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # add data to buffer, a list of step turples
            transitions = []
            for i in range(num_agents):
                transition = (states[i], actions[i], rewards[i], next_states[i], dones[i])
                transitions.append(transition)
                
            buffer.push(transitions)
            
            states = next_states
            scoresPerEpisode += rewards
            
            if len(buffer) > batchsize and i_episode > n_episode_random and t % learn_rate == 0:
                for _ in range(learn_count):
                    samples = buffer.sample(batchsize)
                    for i in range(num_agents):
                        agent.update(samples, i, None)
                    agent.update_targets() #soft update the target network towards the actual networks
                noise *= noise_reduction
            
            if np.any(dones):
                break  

        scorePerEpisode=np.max(scoresPerEpisode)
        if scorePerEpisode > max_score:
            max_score = scorePerEpisode
        scores_deque.append(scorePerEpisode)
        scores_global.append(scorePerEpisode)
        score_average = np.mean(scores_deque)
        scores_average.append(score_average)
        
        tTaken = time.time() - sTime
        totalTimeTaken+=tTaken
        
        #print('\rEpsilon {}\tEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(eps,i_episode, np.mean(scores_deque), score_average), end="")
        if i_episode % 10 == 0:
            print('\r t {} Noise {:.2f} Episode {} buffer {} Score {:.2f}/{:.2f}/{:.2f} Average Score: {:.2f} max_score {:.2f} Time Taken: {:.2f} Total Time Taken: {:.2f}'.format(t, noise, i_episode, len(buffer), scorePerEpisode, scoresPerEpisode[0], scoresPerEpisode[1], score_average, max_score, tTaken, totalTimeTaken))
        if score_average >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, score_average))  
            agent.save_model()
#            break
    return scores_global, scores_average

def train():    
    scores_global, scores_average = maddpgTrain()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores_global)+1), scores_global)
    plt.plot(np.arange(1, len(scores_average)+1), scores_average)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    
def watch_smart_agent():
    global env
    global brain_name
    global brain
    global state_size
    global action_size
    global num_agents
    
    # init agent
    agent = MADDPG(in_actor=state_size, out_actor=action_size, agent_count=num_agents)
    agent.load_model();
    play(random=False, n_episode=5, agent=agent)
    
if __name__ == "__main__":
    loadEnv()
#    play()
    train()
#    watch_smart_agent()