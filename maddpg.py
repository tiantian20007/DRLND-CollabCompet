# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
import numpy as np
from functools import reduce
from utilities import soft_update, transpose_to_tensor, transpose_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("maddpg device:" + device.type)

class MADDPG:
    def __init__(self, in_actor, out_actor, agent_count, critic_obs_full = True, discount_factor=0.99, tau=0.1):
        super(MADDPG, self).__init__()
        # critic input = obs_full + actions = 14+2+2+2=20
        self.agent_count = agent_count
        self.critic_obs_full = critic_obs_full
        if critic_obs_full:
            in_critic = in_actor*agent_count + out_actor*agent_count
        else:
            in_critic = in_actor + out_actor*agent_count
        self.maddpg_agent = []
        for i in range(agent_count):
            self.maddpg_agent.append(DDPGAgent(in_actor, 256, 128, out_actor, in_critic, 256, 128))
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        
    def save_model(self):       
        for i in range(self.agent_count):
            torch.save(self.maddpg_agent[i].actor.state_dict(), 'checkpoint_actor_' + str(i) + '.pth')
            torch.save(self.maddpg_agent[i].critic.state_dict(), 'checkpoint_critic_' + str(i) + '.pth')
            
    def load_model(self):
        for i in range(self.agent_count):
            self.maddpg_agent[i].actor.load_state_dict(torch.load('checkpoint_actor_' + str(i) + '.pth'))
        

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        obs_all_agents = torch.from_numpy(obs_all_agents).float().to(device)
        actions = [agent.act(obs, noise).cpu().data.numpy() for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return np.array(actions)
#        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents """

        states, actions, rewards, next_states, dones = samples[agent_number]
        states_list = []
        actions_list = []
        next_states_list = []
        for i in range(self.agent_count):
            s, a, r, ns, d = samples[i]
            states_list.append(s)
            actions_list.append(a)
            next_states_list.append(ns)
          
        actions_all = reduce((lambda x, y: torch.cat((x, y), dim=1)), actions_list)
        if self.critic_obs_full:
            states_all = reduce((lambda x, y: torch.cat((x, y), dim=1)), states_list)
            next_states_all = reduce((lambda x, y: torch.cat((x, y), dim=1)), next_states_list)
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_states_list)
        target_actions = torch.cat(target_actions, dim=1)
        
        if self.critic_obs_full:
            target_critic_input = torch.cat((next_states_all, target_actions), dim=1).to(device)
        else:
            target_critic_input = torch.cat((next_states, target_actions), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        y = rewards.view(-1, 1) + self.discount_factor * q_next * (1 - dones.view(-1, 1))
#        actions = torch.cat(actions, dim=1)
        if self.critic_obs_full:
            critic_input = torch.cat((states_all, actions_all), dim=1).to(device)
        else:
            critic_input = torch.cat((states, actions_all), dim=1).to(device)
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(states_list) ]
                
        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        if self.critic_obs_full:
            q_input2 = torch.cat((states_all, q_input), dim=1)
        else:
            q_input2 = torch.cat((states, q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        if logger:
            al = actor_loss.cpu().detach().item()
            cl = critic_loss.cpu().detach().item()
            logger.add_scalars('agent%i/losses' % agent_number,
                               {'critic loss': cl,
                                'actor_loss': al},
                               self.iter)

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            




