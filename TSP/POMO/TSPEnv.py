
from dataclasses import dataclass
import torch
import gym
from TSProblemDef import get_random_problems, augment_xy_data_by_8_fold
from gym import spaces
import numpy as np

@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)

class TSPVectorEnv(gym.Env):
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.gamma = env_params['gamma']
        self.batch_size = env_params['batch_size']

        # Const @Load_Problem
        ####################################
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        obs_dict = {
            "observations": spaces.Box(low=0, high=1, shape=(self.problem_size, 2)),
            "action_mask": spaces.MultiBinary([self.batch_size,
                                                self.pomo_size,
                                                self.problem_size]),
            "first_node_idx": spaces.MultiDiscrete([self.problem_size] * self.batch_size * self.pomo_size),
            "last_node_idx": spaces.MultiDiscrete([self.problem_size] * self.batch_size * self.pomo_size),
            "is_initial_action": spaces.Discrete(1)}
        
        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.MultiDiscrete([self.problem_size] * self.batch_size * self.pomo_size)
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

        # self.current_state = None
        # shape: (batch, pomo, node, 5)
        self.hidden_state = None
        # shape: (batch, pomo, hidden_dim)
        
    def reset(self):
        self.visited = np.zeros((self.batch_size, self.pomo_size, self.problem_size), dtype=np.bool)
        self.num_steps = 0
        self.last = np.zeros((self.batch_size, self.pomo_size), dtype=int)
        self.first = np.zeros((self.batch_size, self.pomo_size), dtype=int)
        
        self._generate_random_problems()
        self.state = self._update_state()
        self.info = {}
        self.done = False
        self.total_dist = np.zeros((self.batch_size, self.pomo_size))
        self.total_reward = 0
        return self.state, 0, self.done, self.info, self.problems

    def step(self, selected):
        # selected.shape: (batch, pomo)
        self._go_to(selected)
        self.num_steps += 1
        self.state = self._update_state()
        
        self.done = ((selected == self.first) & self.is_all_visited()).any()
        if ((selected == self.first) & self.is_all_visited()).all() != self.done:
            (selected == self.first) & self.is_all_visited()
        
        reward = self.get_reward(self.done)
        return self.state, reward, self.done, self.info
        
    def _go_to(self, selected):
        # selected.shape: (batch, pomo)
        dest_node_coord = self.get_node_coord(selected)
        # dest_node_coord.shape: (batch, pomo, 2)
        if self.num_steps != 0:
            dist = self.cost(dest_node_coord, self.get_node_coord(self.last))
        else:
            dist = np.zeros((self.batch_size, self.pomo_size))
            self.first = selected
        
        self.last = selected
        
        batch_indices = np.arange(self.batch_size)[:, None]
        pomo_indices = np.arange(self.pomo_size)[None, :]
        self.visited[batch_indices, pomo_indices, selected] = True
        
        self.total_reward = self.total_reward + dist
        
    # def step(self, selected):
    #     # selected.shape: (batch, pomo)

    #     self.selected_count += 1
    #     self.current_node = selected
    #     # shape: (batch, pomo)
    #     self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
    #     # shape: (batch, pomo, 0~problem)

    #     # UPDATE STEP STATE
    #     self.step_state.current_node = self.current_node
    #     # shape: (batch, pomo)
    #     self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
    #     # shape: (batch, pomo, node)

    #     # UPDATE CURRENT STATE
    #     # self.current_state[:, :, :, 4] = 0
    #     # for step in range(self.selected_node_list.size(2)):
    #     #     visited_nodes = self.selected_node_list[:, :, step]
    #     #     self.current_state[self.BATCH_IDX, self.POMO_IDX, visited_nodes, 2] = 0  # 未访问标志设为0
    #     #     self.current_state[self.BATCH_IDX, self.POMO_IDX, visited_nodes, 3] = 1  # 已访问标志设为1
        
    #     # self.current_state[self.BATCH_IDX, self.POMO_IDX, selected, 4] = 1  # 当前节点标志设为1

    #     # returning values
    #     done = (self.selected_count == self.problem_size)
    #     if done:
    #         reward = -self._get_travel_distance()  # note the minus sign!
    #     else:
    #         reward = torch.zeros((self.batch_size, self.pomo_size))

    #     return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def get_reward(self, done):
        if done.any():
            reward = -self.total_dist
            # shape: (batch, pomo)
        else:
            reward = np.zeros((self.batch_size, self.pomo_size))
        return reward
    
    def _update_state(self):
        obs = {"observations": self.problems,
               "action_mask": self._update_mask(),
               "first_node_idx": self.first,
               "last_node_idx": self.last,
               "is_initial_action": self.num_steps == 0,}
        return obs
        
    def _update_mask(self):
        action_mask = ~self.visited
        # shape: (batch, pomo, problem)
        batch_indices = np.arange(self.batch_size)[:, None]
        pomo_indices = np.arange(self.pomo_size)[None, :]
        action_mask[batch_indices, pomo_indices, self.first] |= self.is_all_visited()
        # can only visit first node when all nodes are visited (?)
        return action_mask
        
    def _generate_random_problems(self):
        self.problems = np.random.rand(self.batch_size, self.problem_size, 2)
        
    def is_all_visited(self):
        return self.visited[:, :, :].all(axis=-1)
    
    def cost(self, loc1, loc2):
        # loc1, loc2.shape: (batch, pomo, 2)
        return np.linalg.norm(loc1 - loc2, axis=-1)
    
    def get_node_coord(self, idx):
        # idx.shape: (batch, pomo)
        selected = np.expand_dims(idx, axis=-1)
        # selected.shape: (batch, pomo, 1)
        selected = np.tile(selected, (1, 1, 2))
        # selected.shape: (batch, pomo, 2)
        dest_node_coord = np.take_along_axis(self.problems, selected, axis=1)
        # dest_node_coord.shape: (batch, pomo, 2)
        return dest_node_coord