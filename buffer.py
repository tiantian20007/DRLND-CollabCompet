from collections import namedtuple, deque
import random
import numpy as np
import torch
from utilities import transpose_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self,size,num_agents):
        self.size = size
        self.num_agents = num_agents
        self.deque = deque(maxlen=self.size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    # transitions is a list of tuple
    def push(self, transitions):
        """push into the buffer"""
        
#        input_to_buffer = transpose_list(transition)
#    
#        for item in input_to_buffer:
#            self.deque.append(item)
        
        # each dequeItem is a list of experience
        dequeItem = []
        for i in range(self.num_agents):
            e = self.experience(*transitions[i])
            dequeItem.append(e)
        self.deque.append(dequeItem)
        

    def sample(self, batchsize):
        """sample from the buffer"""
        dequeItems = random.sample(self.deque, batchsize)

        # transpose list of list
#        return transpose_list(samples)
        
        results = []
        for i in range(self.num_agents):
            states = torch.from_numpy(np.vstack([e[i].state for e in dequeItems if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e[i].action for e in dequeItems if e is not None])).float().to(device)
            rewards = torch.from_numpy(np.vstack([e[i].reward for e in dequeItems if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e[i].next_state for e in dequeItems if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack([e[i].done for e in dequeItems if e is not None]).astype(np.uint8)).float().to(device)
            
            results.append((states, actions, rewards, next_states, dones))
        return results

    def __len__(self):
        return len(self.deque)



