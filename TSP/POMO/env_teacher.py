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
        self.gamma = env.gamma

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
        self.n_actions = env.problem_size

    def reset(self):
        self.episode_goal_visited = False
        return np.array(self.env.reset())
    # enddef

    def step(self, action, state_embed, done):
        r_hat = None
        current_state_embed = state_embed.clone().detach()
        next_state, reward, done = self.env.step(action)
        r_hat = self.get_reward(current_state_embed, action, next_state.current_node, done)

        return next_state, r_hat, done

    def get_reward(self, state, action, next_state, done):

        r_orig = self.env.get_reward(done)

        if self.teacher_name == "Orig":
            return r_orig

        elif self.teacher_name == "SelfRS":
            return r_orig + self.Rexploit(state, action) if self.switch_teacher \
                else r_orig

    def get_reward_print(self, state, action):
        next_state = self.env.get_transition(state, action)
        r_hat = self.get_reward(state, action, next_state)
        if torch.is_tensor(r_hat):
            r_hat = float(r_hat.detach().numpy())
        return np.round(r_hat, 4)

    def get_state_int_from_oneHot_encoding(self, state_one_hot):
        state_int = np.nonzero(state_one_hot)[0][0]
        return state_int

    def Rexploit(self, state, action):
        R_exploit = self.SelfRS_network.network(torch.Tensor(state))
        return R_exploit[action] * self.clipping_epsilon
    # enddef

    def update(self, D, agent=None, epsilon_reinforce=0.0):
        return self.update_SelfRS(D)

    def update_SelfRS(self, D):

        self.first_succesfull_traj = True
        torch.autograd.set_detect_anomaly(True)
        postprocess_D = self.postprocess_data(D)

        for traj in postprocess_D:
            states_batch = []
            returns_batch_G_bar = []
            accumulator = []

            if traj[0]['G_bar'][0, 0] > 0.0 and self.first_succesfull_traj:
                # 直接取第一个batch的第一个pomo的G_bar
                self.nonzero_return_count += 1
                self.first_succesfull_traj = False

            for step in traj:

                # save states batch and returns batch for training value network
                s = step['state']
                a = step['action']
                probs = step['probs']
                prob = step['prob']
                G_bar = step['G_bar']
                states_batch.append(s)
                returns_batch_G_bar.append(G_bar)
                V_s = self.value_network.network(torch.Tensor(s)).detach().squeeze()
                one_hot_encoding_action_a = np.zeros(self.n_actions)
                one_hot_encoding_action_a[a.cpu()] = 1.0
                one_hot_encoding_action_a_var = Variable(torch.Tensor(one_hot_encoding_action_a))

                # sum over action b
                accumulator_sum_action_b = []
                for b in range(self.n_actions):
                    one_hot_encoding_action_b = np.zeros(self.n_actions)
                    one_hot_encoding_action_b[b] = 1.0
                    one_hot_encoding_action_b_var = Variable(torch.Tensor(one_hot_encoding_action_b))
                    accumulator_sum_action_b.append(torch.sum(self.SelfRS_network.network(torch.Tensor(s)) *
                                                              one_hot_encoding_action_b_var * probs[b]))

                final_result_left_hand_side = torch.sum(self.SelfRS_network.network(torch.Tensor(s)) *
                                                        one_hot_encoding_action_a_var) \
                                              - torch.sum(torch.stack(accumulator_sum_action_b))

                accumulator.append(prob * (G_bar - V_s) *
                                   final_result_left_hand_side)
            
            loss = -torch.mean(torch.stack(accumulator))
            # print(f"Loss shape: {loss.shape}")
            # print(f"Loss computation graph:", loss.grad_fn)
            # print(f"Loss requires_grad: {loss.requires_grad}")

            # for name, param in self.SelfRS_network.named_parameters():
            #     print(f"Parameter {name}:")
            #     print(f"  requires_grad: {param.requires_grad}")
            #     print(f"  shape: {param.shape}")
            # for name, param in self.SelfRS_network.named_parameters():
            #     if param.grad is not None:
            #         print(f"Gradient for {name}:", param.grad.norm().item())
            #     else:
            #         print(f"No gradient for {name} - Parameter shape:", param.shape)
            # update SelfRS network
            self.SelfRS_network.zero_grad()
            self.SelfRS_network.optimizer.zero_grad()
            
            loss.backward()
            self.SelfRS_network.optimizer.step()

            print(type(states_batch))
            states_batch = [s.detach().cpu() if torch.is_tensor(s) else s for s in states_batch]
            states_batch = np.array(states_batch)
            returns_batch_G_bar = [G_bar.detach().cpu() if torch.is_tensor(G_bar) else G_bar for G_bar in returns_batch_G_bar]
            returns_batch_G_bar = np.array(returns_batch_G_bar)

            self.update_value_network(states_batch, returns_batch_G_bar)

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
            e_t = {
                'state': episode[t]['state'],
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

    def get_original_reward(self, env_orig, done):
        r_bar = env_orig.get_reward(done)
        return r_bar

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
