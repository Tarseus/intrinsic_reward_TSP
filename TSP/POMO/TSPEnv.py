import gym
import numpy as np
from gym import spaces
import TSPModel
import torch

def assign_env_config(self, kwargs):
    """
    Set self.key = value, for each key in kwargs
    """
    for key, value in kwargs.items():
        setattr(self, key, value)


def dist(loc1, loc2):
    return ((loc1[:, 0] - loc2[:, 0]) ** 2 + (loc1[:, 1] - loc2[:, 1]) ** 2) ** 0.5

from TSPModel import RexploitNetwork, CriticNetwork

class SharedSelfRSNetwork:
    _instance = None

    @staticmethod
    def get_instance(model_params):
        if SharedSelfRSNetwork._instance is None:
            SharedSelfRSNetwork._instance = RexploitNetwork(**model_params)
        return SharedSelfRSNetwork._instance
    
class SharedValueNetwork:
    _instance = None

    @staticmethod
    def get_instance(model_params):
        if SharedValueNetwork._instance is None:
            SharedValueNetwork._instance = CriticNetwork(**model_params)
        return SharedValueNetwork._instance

class TSPVectorEnv(gym.Env):
    def __init__(self, model_params, trainer_params, **env_params):
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.gamma = env_params['gamma']
        self.reward_type = env_params['reward_type']

        obs_dict = {"observations": spaces.Box(low=0, high=1, shape=(self.problem_size, 2))}
        obs_dict["ninf_mask"] = spaces.Box(
            low=-np.inf, high=0, shape=(self.pomo_size, self.problem_size)
        )
        obs_dict["first_node_idx"] = spaces.MultiDiscrete([self.problem_size] * self.pomo_size)
        obs_dict["last_node_idx"] = spaces.MultiDiscrete([self.problem_size] * self.pomo_size)
        obs_dict["is_initial_action"] = spaces.Discrete(1)

        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.MultiDiscrete([self.problem_size] * self.pomo_size)
        self.reward_space = None
        self.total_dist = 0

        self.reset()
        
        self.model_params = model_params
        self.trainer_params = trainer_params
        self.SelfRS_network = SharedSelfRSNetwork.get_instance(model_params)
        self.value_network = SharedValueNetwork.get_instance(model_params)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.visited = np.zeros((self.pomo_size, self.problem_size), dtype=bool)
        self.num_steps = 0
        self.last = np.zeros(self.pomo_size, dtype=int)  # idx of the first elem
        self.first = np.zeros(self.pomo_size, dtype=int)  # idx of the first elem

        self._generate_problems()
        self.state = self._update_state()
        self.info = {}
        self.done = False
        self.total_dist = 0
        return self.state

    def _generate_problems(self):
        self.problems = np.random.rand(self.problem_size, 2)

    def step(self, action):

        self._go_to(action)  # Go to node 'action', modify the reward
        self.num_steps += 1
        self.state = self._update_state()

        # need to revisit the first node after visited all other nodes
        self.done = (action == self.first) & self.is_all_visited()
        assert self.done.any() == self.done.all() # only one done at a time

        if self.done.any():
            self.reward = -self.total_dist
        else:
            self.reward = np.zeros(self.pomo_size)

        return self.state, self.reward, self.done, self.info

    def total_reward_step(self, action, state_dict, decoder_q_first):
        next_state, reward, done, info = self.step(action)
        self.SelfRS_network.q_first = decoder_q_first
        self.SelfRS_network.q_first_steps = decoder_q_first[:, :, 0, :]
        self.value_network.q_first = decoder_q_first[:, :, 0, :]
        r_hat = self.get_total_reward(state_dict, action, next_state, done)
        
        return next_state, r_hat, done

    # Euclidean cost function
    def cost(self, loc1, loc2):
        return dist(loc1, loc2)

    def is_all_visited(self):
        # assumes no repetition in the first `problem_size` steps
        return self.visited[:, :].all(axis=1)

    def _go_to(self, destination):
        dest_node = self.problems[destination]
        if self.num_steps != 0:
            dist = self.cost(dest_node, self.problems[self.last])
        else:
            dist = np.zeros(self.pomo_size)
            self.first = destination

        self.last = destination

        self.visited[np.arange(self.pomo_size), destination] = True
        
        self.total_dist = self.total_dist + dist

    def _update_state(self):
        obs = {"observations": self.problems}  # n x 2 array
        obs["ninf_mask"] = self._update_mask()
        obs["first_node_idx"] = self.first
        obs["last_node_idx"] = self.last
        obs["is_initial_action"] = self.num_steps == 0
        return obs

    def _update_mask(self):
        # Only allow to visit unvisited nodes
        action_mask = ~self.visited
        # can only visit first node when all nodes have been visited
        action_mask[np.arange(self.pomo_size), self.first] |= self.is_all_visited()
        action_mask = np.where(action_mask, 0, -np.inf)
        return action_mask

    def get_total_reward(self, state_dict, action, next_state, done):
        r_orig = self.reward
        if self.reward_type == 'SelfRS':
            return r_orig + self.Rexploit(state_dict, action)
        elif self.reward_type == 'Original':
            return r_orig
        
    def Rexploit(self, state, action):
        # action.shape: (batch, pomo)
        state_embedding = state['embed_node']
        ninf_mask = state['ninf_mask']
        R_exploit = self.SelfRS_network.batch_forward(state_embedding, ninf_mask)
        # shape: (batch, pomo, problem_size)
        batch_size, pomo_size, problem_size = R_exploit.shape
        batch_indices = np.arange(batch_size)[:, None]  # shape: (batch, 1)
        pomo_indices = np.arange(pomo_size)  # shape: (pomo,)

        selected_R_exploit = R_exploit[batch_indices, pomo_indices, action]
        return selected_R_exploit * self.clipping_epsilon
    
    def update(self, buffer):
        if self.reward_type == 'SelfRS':
            self.update_SelfRS(buffer)
            
    def update_SelfRS(self, buffer):
        postprocess_D = self.postprocess_data(buffer)
        recent_buffer_size = self.trainer_params['reward_update_freq']
        for traj in postprocess_D[-recent_buffer_size:]:
            # states_batch = []
            # returns_batch_G_bar = []
            accumulator = []

            if traj[0]['G_bar'][0, 0] > 0.0 and self.first_succesfull_traj:
                # 直接取第一个batch的第一个pomo的G_bar
                self.nonzero_return_count += 1
                self.first_succesfull_traj = False

            s_batch = torch.stack([step['state_dict']['embed_node'][:, 0, :].detach() for step in traj]) # shape: (steps, batch, state_dim)
            ninf_mask_batch = torch.stack([step['state_dict']['ninf_mask'][:, 0, :].detach() for step in traj]) # shape: (steps, batch, problem_size)
            # print(f"ninf_mask_batch: {ninf_mask_batch}")
            a_batch = torch.stack([step['action'][:, 0].detach() for step in traj]) # shape: (steps, batch)
            probs_batch = torch.stack([step['probs'][:, 0, :].detach() for step in traj]) # shape: (steps, batch, problem_size)
            prob_batch = torch.stack([step['prob'][:, 0].detach() for step in traj]) # shape: (steps, batch)
            G_bar_batch = torch.stack([step['G_bar'][:, 0].detach() for step in traj]) # shape: (steps, batch)
            V_s_batch = self.value_network(s_batch, ninf_mask_batch).squeeze() # shape: (steps, batch)
            selected_values_batch = self.SelfRS_network.steps_forward(s_batch, ninf_mask_batch)  # shape: (steps, batch, problem_size)
            base_batch = torch.sum(selected_values_batch * probs_batch, dim=2) # shape: (steps, batch)

            a_batch_expanded = a_batch.unsqueeze(-1)  # shape: (steps, batch, 1)
            selected_value_a_batch = torch.gather(selected_values_batch, 2, a_batch_expanded).squeeze(-1)  # shape: (steps, batch)
            final_result_left_hand_side_batch = selected_value_a_batch - base_batch  # shape: (steps, batch)
            accumulator = prob_batch * (G_bar_batch - V_s_batch) * final_result_left_hand_side_batch  # shape: (steps, batch)
                
            loss = -torch.mean(accumulator)
            # update SelfRS network
            self.SelfRS_network.zero_grad()
            self.SelfRS_network.optimizer.zero_grad()
            
            loss.backward()
            self.SelfRS_network.optimizer.step()

            self.update_value_network(s_batch, G_bar_batch, ninf_mask_batch)
            
    def update_value_network(self, states_batch, returns_batch, ninf_mask_batch = None):
        # update value network
        loss_critic = self.value_network.update(states_batch, returns_batch, ninf_mask_batch)
        
    def postprocess_data(self, D):
        postprocessed_D = []

        for episode in D:
            postprocessed_epidata = self.get_postposessed_episode(self.env, episode)

            # add postprocessed episode
            postprocessed_D.append(postprocessed_epidata)

        return postprocessed_D
    
    def get_postposessed_episode(self, env_orig, episode):

        postprocessed_epidata = []
        for t in range(len(episode)):
            # get original reward
            done = episode[t]['done']
            r_bar = self.get_original_reward(env_orig, done)
            if isinstance(r_bar, np.ndarray):
                r_bar = torch.tensor(r_bar).float()
            e_t = {
                'state_dict': episode[t]['state_dict'],
                'action': episode[t]['action'],
                'reward_hat': episode[t]['reward_hat'],
                'G_hat': episode[t]['G_hat'],
                'prob': episode[t]['prob'],
                'probs': episode[t]['probs'],
                'done': done,
                'reward_bar': r_bar,
                'G_bar': None,
            }
            postprocessed_epidata.append(e_t)

        G_bar = torch.zeros_like(episode[-1]['reward_hat'])
        for i in range(len(postprocessed_epidata) - 1, -1, -1):
            reward = postprocessed_epidata[i]['reward_bar']
            G_bar = reward + self.env.gamma * G_bar
            postprocessed_epidata[i]['G_bar'] = G_bar.detach()

        return postprocessed_epidata
    
    