import numpy as np
import torch
from torch.autograd import Variable
import gym
import copy
import utils
import models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EnvTeacher(gym.Env):

    def __init__(self, env, args, teacher_name):
        super(EnvTeacher, self).__init__()
        self.teachers = ["Orig", "ExploB", "SelfRS", "ExploRS", "sors", "SORS_with_Rbar", "LIRPG_without_metagrad"]
        if teacher_name not in self.teachers:
            print("Error!!!")
            print("Teacher name should be one of the following: {}".format(self.teachers))
            exit(0)
        self.env = env
        self.args = args
        self.teacher_name = teacher_name

        # declare open gym necessary attributes
        # self.observation_space = self.env.observation_space
        # self.action_space = self.env.action_space
        # self.n_actions = env.n_actions
        # self.gamma = env.gamma

        # discretization
        self.chunk_size = 0.1
        self.n_chunks = int(1 / self.chunk_size)
        # self.ExploB_w = np.zeros(2 * 2 * 2 * self.n_chunks)
        # self.ExploB_w = np.zeros((1 + env.n_picks) * self.n_chunks) # has key or does not have key state

        # self.chunk_centroids = self.get_chunks_zero_one_interval()

        # declate different type of teacher's networks
        self.SelfRS_network = models.RexploitNetwork(env, args) #.to(device)
        self.value_network = models.CriticNetwork(env, args) #.to(device)
        # self.rsors_network = models.RSORSNetwork(env, args) #.to(device)
        # self.lirpg_network = models.RLIRPGNetwork(env, args) #.to(device)

        self.goal_visits = 0.0
        self.episode_goal_visited = None
        self.nonzero_return_count = 0
        self.n_minimal_network_updates = 100
        self.nonzero_return_count = 0
        self.first_succesfull_traj = None
        self.clipping_epsilon = args["clipping_epsilon"]
        self.n_steps_to_follow_orig = 5000 #2000
        self.switch_teacher = False
    # enddef

    def reset(self):
        self.episode_goal_visited = False
        return np.array(self.env.reset())
    # enddef

    def step(self, action, state_embed, done):
        r_hat = None
        current_state_embed = state_embed.clone().detach()
        next_state, reward, done = self.env.step(action)
        r_hat = self.get_reward(current_state_embed, action, next_state.current_node, done)

        # if (self.env.n_picks == 1):
        #     if (self.env.reward_range[0] <= current_state[0]
        #         <= self.env.reward_range[1] and current_state[1] == 1):
        #         self.goal_visits += 1.0

        # if (self.env.n_picks > 1):
        #     if (self.env.reward_range[0] <= current_state[0]
        #         <= self.env.reward_range[1] and current_state[4] == 1):
        #         self.goal_visits += 1.0

        # return np.array(next_state.current_node.cpu()), r_hat, done
        return next_state, r_hat, done
    # enddef

    def get_reward(self, state, action, next_state, done):

        r_orig = self.env.get_reward(done)

        if self.teacher_name == "Orig":
            return r_orig

        elif self.teacher_name == "SelfRS":
            return r_orig + self.Rexploit(state, action) if self.switch_teacher \
                else r_orig

        elif self.teacher_name == "ExploRS":
            #update R_explore
            self.update_Rexplore_given_state(state)

            return r_orig + self.Rexploit(state, action) + self. Rexplore(next_state) if self.switch_teacher \
                else r_orig + self.Rexplore(next_state)

    def get_reward_print(self, state, action):
        next_state = self.env.get_transition(state, action)
        r_hat = self.get_reward(state, action, next_state)
        if torch.is_tensor(r_hat):
            r_hat = float(r_hat.detach().numpy())
        return np.round(r_hat, 4)
    #enddef

    def get_state_int_from_oneHot_encoding(self, state_one_hot):
        state_int = np.nonzero(state_one_hot)[0][0]
        return state_int
    #enddef

    def Rexplore(self, state):

        if self.env.terminal_state == 1 and \
                state[0] == - 1:
            return 0.0

        numerator = self.args["ExploB_lmbd"]
        N_s = np.power(self.args["ExploB_lmbd"] / self.args["ExploB_max"], 2.0) + self.ExploB_w[self.phi(state)]
        denominator = np.sqrt(N_s)
        # print(numerator/denominator)
        return numerator/denominator
    # enddef

    def Rexploit(self, state, action):
        R_exploit = self.SelfRS_network.network(torch.Tensor(state))
        return R_exploit[action] * self.clipping_epsilon
    # enddef

    def R_ExploRS(self, state, action, next_state):
        return self.Rexploit(state, action) + self.Rexplore(next_state)
    #enddef

    def update(self, D, agent=None, epsilon_reinforce=0.0):
        return self.update_ExploRS(D)

    # def update_Rexplore_given_state(self, state):
    #     self.ExploB_w[self.phi(state)] += 1.0
    #enddef

    def phi(self, state):

        coord_x = state[0]
        chunk_number = 0
        if self.env.n_picks == 1:
            if state[1] == 1:
                chunk_number = 1

        if self.env.n_picks > 1:
            if state[1] == 1:
                chunk_number = 1 + np.nonzero(state[4:])[0][0]


        abstracted_state_tmp = np.floor(coord_x / self.chunk_size)

        if abstracted_state_tmp == self.n_chunks:
            abstracted_state_tmp = self.n_chunks - 1

        abstracted_state = int(((chunk_number) * self.n_chunks) + abstracted_state_tmp)

        return abstracted_state
    #enddef

    def update_SelfRS(self, D):

        self.first_succesfull_traj = True

        postprocess_D = self.postprocess_data(D)

        for traj in postprocess_D:
            states_batch = []
            returns_batch_G_bar = []
            accumulator = []

            if traj[0][7] > 0.0 and self.first_succesfull_traj:
                self.nonzero_return_count += 1
                self.first_succesfull_traj = False

            for s, a, _, _, _, pi_given_s_array, _, G_bar in traj:

                # save states batch and returns batch for training value network
                states_batch.append(s)
                returns_batch_G_bar.append(G_bar)
                V_s = float(self.value_network.network(torch.Tensor(s)).detach().numpy())
                one_hot_encoding_action_a = np.zeros(self.n_actions)
                one_hot_encoding_action_a[a] = 1.0
                one_hot_encoding_action_a_var = Variable(torch.Tensor(one_hot_encoding_action_a))

                # sum over action b
                accumulator_sum_action_b = []
                for b in range(self.n_actions):
                    one_hot_encoding_action_b = np.zeros(self.n_actions)
                    one_hot_encoding_action_b[b] = 1.0
                    one_hot_encoding_action_b_var = Variable(torch.Tensor(one_hot_encoding_action_b))
                    accumulator_sum_action_b.append(torch.sum(self.SelfRS_network.network(torch.Tensor(s)) *
                                                              one_hot_encoding_action_b_var * pi_given_s_array[b]))

                final_result_left_hand_side = torch.sum(self.SelfRS_network.network(torch.Tensor(s)) *
                                                        one_hot_encoding_action_a_var) \
                                              - torch.sum(torch.stack(accumulator_sum_action_b))

                accumulator.append(pi_given_s_array[a] * (G_bar - V_s) *
                                   final_result_left_hand_side)

            loss = -torch.mean(torch.stack(accumulator))

            # update SelfRS network
            self.SelfRS_network.zero_grad()
            self.SelfRS_network.optimizer.zero_grad()
            loss.backward()
            self.SelfRS_network.optimizer.step()

            states_batch = np.array(states_batch)
            returns_batch_G_bar = np.array(returns_batch_G_bar)

            self.update_value_network(states_batch, returns_batch_G_bar)

    # enddef

    def update_value_network(self, states_batch, returns_batch):
        # update value network
        loss_critic = self.value_network.update(states_batch, returns_batch)
    #enddef

    def update_ExploRS(self, D):
        # Update ExploB
        # self.update_ExploB(D)

        # Update SelfRS
        self.update_SelfRS(D)
    #enddef

    def compute_df_dtheta(self, pi_log_softmax, actor_network):
        actor_network.zero_grad()

        grad_theta_pi = torch.autograd.grad(pi_log_softmax, actor_network.parameters(), retain_graph=True)
        grad_theta_pi = [item.view(-1) for item in grad_theta_pi]
        grad_theta_pi = torch.cat(grad_theta_pi)

        return grad_theta_pi
    # enddef

    def postprocess_data(self, D):

        postprocessed_D = []

        # episode --> [state, action, \hat{r}, next_state, \hat(G), \pi(.|state)]
        for episode in D:
            # postProcessed episode --> [state, action, \hat{r}, next_state, \hat(G), \pi(.|state), \bar{r}, \bar{G}]
            postprocessed_epidata = self.get_postposessed_episode(self.env, episode)

            # add postprocessed episode
            postprocessed_D.append(postprocessed_epidata)

        return postprocessed_D
    #enddef

    def get_postposessed_episode(self, env_orig, episode):

        postprocessed_epidata = []
        for t in range(len(episode)):
            state, action, r_hat, next_state, G_hat, pi_given_s = episode[t]

            # get original reward
            r_bar = self.get_original_reward(env_orig, state, action, next_state)

            # postProcessed episode --> [state, action, \hat{r}, next_state, \hat(G), \pi(.|state), \bar{r}, \bar{G}]
            e_t = [state, action, r_hat, next_state, G_hat, pi_given_s, r_bar, 0.0]

            postprocessed_epidata.append(e_t)

        # compute return \bar{G} for every (s, a)
        G_bar = 0  # original return
        for i in range(len(postprocessed_epidata) - 1, -1, -1):  # iterate backwards
            _, _, _, _, _, _, r_bar, _ = postprocessed_epidata[i]
            G_bar = r_bar + env_orig.gamma * G_bar
            postprocessed_epidata[i][7] = G_bar  # update G_bar in episode

        return postprocessed_epidata
    #enddef

    def get_original_reward(self, env_orig, state, action, next_state=None):
        r_bar = env_orig.get_reward(state, action)
        return r_bar
    #enddef

    def get_pairwise_data_using_return(self, postprocessed_D):
        pairwise_data = []

        for i, episode_i in enumerate(postprocessed_D):
            for j, episode_j in enumerate(postprocessed_D):
                G_bar_i = episode_i[0][7]
                G_bar_j = episode_j[0][7]
                if G_bar_i > G_bar_j:
                    # \tau_i > \tau_j
                    pairwise_data.append([i, j, 1.0])

        return pairwise_data
    # enndef

    def softmax_prob(self, a, b):
        return np.exp(a) / (np.exp(a) + np.exp(b))
    #enddef

    def indicator(self, state, action, s, a):
        return 1.0 if state == s and action == a else 0.0
# endclass
