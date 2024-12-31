import numpy as np
import torch
from torch.autograd import Variable
import gym
import copy
import utils
import TSPModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EnvTeacher(gym.Env):

    def __init__(self, env, args, teacher_name, model_params, trainer_params):
        super(EnvTeacher, self).__init__()
        self.teachers = ["Orig", "ExploB", "SelfRS", "ExploRS", "sors", "SORS_with_Rbar", "LIRPG_without_metagrad"]
        if teacher_name not in self.teachers:
            print("Error!!!")
            print("Teacher name should be one of the following: {}".format(self.teachers))
            exit(0)
        self.env = env
        self.args = args
        self.trainer_params = trainer_params
        self.teacher_name = teacher_name

        # declare open gym necessary attributes
        # self.observation_space = self.env.observation_space
        # self.action_space = self.env.action_space
        # self.n_actions = env.n_actions
        self.gamma = env.gamma

        # discretization
        self.chunk_size = 0.1
        self.n_chunks = int(1 / self.chunk_size)

        # declate different type of teacher's networks
        self.SelfRS_network = TSPModel.RexploitNetwork(**model_params) #.to(device)
        self.value_network = TSPModel.CriticNetwork(**model_params) #.to(device)

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

    def step(self, action, state_dict, done):
        r_hat = None
        current_state_embed = state_dict['embed_node']
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

    def update(self, D, decoder_q_first):
        return self.update_SelfRS(D, decoder_q_first)

    def update_SelfRS(self, D, decoder_q_first):
        # print(f"Allocated memory(before teacher update): {torch.cuda.memory_allocated() / 1024**2} MB")
        # print(f"Reserved memory(before teacher update): {torch.cuda.memory_reserved() / 1024**2} MB")
        self.first_succesfull_traj = True
        self.SelfRS_network.q_first = decoder_q_first
        self.value_network.q_first = decoder_q_first
        postprocess_D = self.postprocess_data(D)
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
            
            selected_values_batch = self.SelfRS_network(s_batch, ninf_mask_batch)  # shape: (steps, batch, problem_size)
            base_batch = torch.sum(selected_values_batch * probs_batch, dim=2) # shape: (steps, batch)
            
            a_batch_expanded = a_batch.unsqueeze(-1)  # shape: (steps, batch, 1)
            selected_value_a_batch = torch.gather(selected_values_batch, 2, a_batch_expanded).squeeze(-1)  # shape: (steps, batch)
            final_result_left_hand_side_batch = selected_value_a_batch - base_batch  # shape: (steps, batch)
            accumulator = prob_batch * (G_bar_batch - V_s_batch) * final_result_left_hand_side_batch  # shape: (steps, batch)

            # 检查 accumulator 的值
            # print(f"accumulator: {accumulator}")
            # del a_batch, probs_batch, prob_batch, V_s_batch, selected_values_batch, base_batch, selected_value_a_batch, final_result_left_hand_side_batch
            # torch.cuda.empty_cache()
                
            loss = -torch.mean(accumulator)
            # update SelfRS network
            self.SelfRS_network.zero_grad()
            self.SelfRS_network.optimizer.zero_grad()
            
            loss.backward()
            self.SelfRS_network.optimizer.step()

            self.update_value_network(s_batch, G_bar_batch, ninf_mask_batch)
            # del s_batch, G_bar_batch, accumulator
            # torch.cuda.empty_cache()
        # print(f"Allocated memory(after teacher update): {torch.cuda.memory_allocated() / 1024**2} MB")
        # print(f"Reserved memory(after teacher update): {torch.cuda.memory_reserved() / 1024**2} MB")

    def update_value_network(self, states_batch, returns_batch, ninf_mask_batch = None):
        # update value network
        loss_critic = self.value_network.update(states_batch, returns_batch, ninf_mask_batch)
    #enddef

    def update_ExploRS(self, D):
        # Update ExploB
        # self.update_ExploB(D)

        # Update SelfRS
        self.update_SelfRS(D)
    #enddef

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
