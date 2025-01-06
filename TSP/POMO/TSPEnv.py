import gym
import numpy as np
from gym import spaces

def assign_env_config(self, kwargs):
    """
    Set self.key = value, for each key in kwargs
    """
    for key, value in kwargs.items():
        setattr(self, key, value)


def dist(loc1, loc2):
    return ((loc1[:, 0] - loc2[:, 0]) ** 2 + (loc1[:, 1] - loc2[:, 1]) ** 2) ** 0.5


class TSPVectorEnv(gym.Env):
    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.gamma = env_params['gamma']

        obs_dict = {"observations": spaces.Box(low=0, high=1, shape=(self.problem_size, 2))}
        obs_dict["action_mask"] = spaces.MultiBinary(
            [self.pomo_size, self.problem_size]
        )  # 1: OK, 0: cannot go
        obs_dict["first_node_idx"] = spaces.MultiDiscrete([self.problem_size] * self.pomo_size)
        obs_dict["last_node_idx"] = spaces.MultiDiscrete([self.problem_size] * self.pomo_size)
        obs_dict["is_initial_action"] = spaces.Discrete(1)

        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.MultiDiscrete([self.problem_size] * self.pomo_size)
        self.reward_space = None
        self.total_dist = 0

        self.reset()

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
        return self.state, 0, self.done, self.info, self.problems

    def _generate_problems(self):
        self.problems = np.random.rand(self.problem_size, 2)

    def step(self, action):

        self._go_to(action)  # Go to node 'action', modify the reward
        self.num_steps += 1
        self.state = self._update_state()

        # need to revisit the first node after visited all other nodes
        self.done = (action == self.first) & self.is_all_visited()
        if self.done:
            self.reward = -self.total_dist
        else:
            self.reward = np.zeros(self.pomo_size)

        return self.state, self.reward, self.done, self.info

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
        obs["action_mask"] = self._update_mask()
        obs["first_node_idx"] = self.first
        obs["last_node_idx"] = self.last
        obs["is_initial_action"] = self.num_steps == 0
        return obs

    def _update_mask(self):
        # Only allow to visit unvisited nodes
        action_mask = ~self.visited
        # can only visit first node when all nodes have been visited
        action_mask[np.arange(self.pomo_size), self.first] |= self.is_all_visited()
        return action_mask
